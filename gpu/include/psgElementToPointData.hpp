#pragma once

#include <lsMesh.hpp>

#include <rayParticle.hpp>
#include <raygIndexMap.hpp>

#include <vcCudaBuffer.hpp>
#include <vcKDTree.hpp>

#include <algorithm>
#include <cassert>
#include <numeric>
#include <stdexcept>

namespace viennaps::gpu {

using namespace viennacore;

template <typename NumericType, typename MeshNT = NumericType>
class ElementToPointData {
  CudaBuffer &d_elementData_;
  SmartPointer<viennals::PointData<NumericType>> pointData_;
  const viennaray::gpu::IndexMap indexMap_;
  SmartPointer<KDTree<NumericType, Vec3D<NumericType>>> elementKdTree_;
  SmartPointer<viennals::Mesh<NumericType>> diskMesh_;
  SmartPointer<viennals::Mesh<MeshNT>> surfaceMesh_;
  const NumericType conversionRadius_;

  static constexpr bool discard2 = true;
  static constexpr bool discard4 = true;

public:
  ElementToPointData(
      CudaBuffer &d_elementData,
      SmartPointer<viennals::PointData<NumericType>> pointData,
      const std::vector<viennaray::gpu::Particle<NumericType>> &particles,
      SmartPointer<KDTree<NumericType, Vec3D<NumericType>>> elementKdTree,
      SmartPointer<viennals::Mesh<NumericType>> diskMesh,
      SmartPointer<viennals::Mesh<MeshNT>> surfMesh,
      const NumericType conversionRadius)
      : d_elementData_(d_elementData), pointData_(pointData),
        indexMap_(particles), elementKdTree_(elementKdTree),
        diskMesh_(diskMesh), surfaceMesh_(surfMesh),
        conversionRadius_(conversionRadius) {}

  ElementToPointData(
      CudaBuffer &d_elementData,
      SmartPointer<viennals::PointData<NumericType>> pointData,
      const viennaray::gpu::IndexMap &indexMap,
      SmartPointer<KDTree<NumericType, Vec3D<NumericType>>> elementKdTree,
      SmartPointer<viennals::Mesh<NumericType>> diskMesh,
      SmartPointer<viennals::Mesh<MeshNT>> surfMesh,
      const NumericType conversionRadius)
      : d_elementData_(d_elementData), pointData_(pointData),
        indexMap_(indexMap), elementKdTree_(elementKdTree), diskMesh_(diskMesh),
        surfaceMesh_(surfMesh), conversionRadius_(conversionRadius) {}

  void apply() {

    const auto numData = indexMap_.getNumberOfData();
    const auto &points = diskMesh_->nodes;
    const auto numPoints = points.size();
    const auto numElements = elementKdTree_->getNumberOfPoints();
    const auto normals = diskMesh_->cellData.getVectorData("Normals");
    const auto elementNormals = surfaceMesh_->cellData.getVectorData("Normals");
    const auto sqrdDist = conversionRadius_ * conversionRadius_;

    // retrieve data from device
    std::vector<MeshNT> elementData(numData * numElements);
    d_elementData_.download(elementData.data(), numData * numElements);

    // prepare point data container
    pointData_->clear();
    for (const auto &label : indexMap_) {
      std::vector<NumericType> data(numPoints, 0.);
      pointData_->insertNextScalarData(std::move(data), label);
    }

    if (pointData_->getScalarDataSize() != numData) {
      Logger::getInstance()
          .addError("ElementToPointData: "
                    "Number of data arrays does not match expected count")
          .print();
    }

#pragma omp parallel for schedule(static)
    for (unsigned i = 0; i < numPoints; i++) {

      // we have to use the squared distance here
      auto closePoints =
          elementKdTree_->findNearestWithinRadius(points[i], sqrdDist).value();

      std::vector<NumericType> weights(closePoints.size(), 0.);

      unsigned numClosePoints = 0;
      for (std::size_t n = 0; n < closePoints.size(); ++n) {
        const auto &p = closePoints[n];
        assert(p.first < numElements);

        const auto weight =
            DotProductNT(normals->at(i), elementNormals->at(p.first));

        if (weight > NumericType(1e-6) && !std::isnan(weight)) {
          weights[n] = weight;
          ++numClosePoints;
        }
      }

      std::size_t nearestIdx = 0;
      if (numClosePoints == 0) { // fallback to nearest point
        auto nearestPoint = elementKdTree_->findNearest(points[i]);
        nearestIdx = nearestPoint->first;
      }

      for (unsigned j = 0; j < numData; ++j) {
        NumericType value = NumericType(0);

        if (numClosePoints > 0) {
          auto weightsCopy = weights;

          // Discard outliers if enabled
          discardOutliers(weightsCopy, weights, closePoints, elementData, j,
                          numElements, numClosePoints);

          // Compute weighted average
          const NumericType sum = std::accumulate(
              weightsCopy.begin(), weightsCopy.end(), NumericType(0));

          if (sum > NumericType(0)) {
            for (std::size_t k = 0; k < closePoints.size(); ++k) {
              if (weightsCopy[k] > NumericType(0)) {
                const unsigned elementIdx =
                    closePoints[k].first + j * numElements;
                value += weightsCopy[k] * elementData[elementIdx];
              }
            }
            value /= sum;
          } else {
            // Fallback if all weights were discarded
            auto nearestPoint = elementKdTree_->findNearest(points[i]);
            value = elementData[nearestPoint->first + j * numElements];
          }
        } else {
          // Fallback to nearest point
          value = elementData[nearestIdx + j * numElements];
        }

        pointData_->getScalarData(j)->at(i) = value;
      }
    }
  }

private:
  // Helper function to find min/max values and their indices
  struct MinMaxInfo {
    NumericType min1, min2, max1, max2;
    int minIdx1, minIdx2, maxIdx1, maxIdx2;

    MinMaxInfo()
        : min1(std::numeric_limits<NumericType>::max()),
          min2(std::numeric_limits<NumericType>::max()), max1(0), max2(0),
          minIdx1(-1), minIdx2(-1), maxIdx1(-1), maxIdx2(-1) {}
  };

  MinMaxInfo findMinMaxValues(
      const std::vector<NumericType> &weights,
      const std::vector<std::pair<std::size_t, NumericType>> &closePoints,
      const std::vector<MeshNT> &elementData, unsigned j,
      unsigned numElements) {
    MinMaxInfo info;

    for (std::size_t k = 0; k < closePoints.size(); ++k) {
      if (weights[k] != NumericType(0)) {
        const unsigned elementIdx = closePoints[k].first + j * numElements;
        const auto value = elementData[elementIdx];

        // Update min values
        if (value < info.min1) {
          info.min2 = info.min1;
          info.minIdx2 = info.minIdx1;
          info.min1 = value;
          info.minIdx1 = static_cast<int>(k);
        } else if (value < info.min2) {
          info.min2 = value;
          info.minIdx2 = static_cast<int>(k);
        }

        // Update max values
        if (value > info.max1) {
          info.max2 = info.max1;
          info.maxIdx2 = info.maxIdx1;
          info.max1 = value;
          info.maxIdx1 = static_cast<int>(k);
        } else if (value > info.max2) {
          info.max2 = value;
          info.maxIdx2 = static_cast<int>(k);
        }
      }
    }

    return info;
  }

  void discardOutliers(
      std::vector<NumericType> &weightsCopy,
      const std::vector<NumericType> &weights,
      const std::vector<std::pair<std::size_t, NumericType>> &closePoints,
      const std::vector<MeshNT> &elementData, unsigned j, unsigned numElements,
      unsigned numClosePoints) {

    if (discard4 && numClosePoints > 4) {
      const auto info =
          findMinMaxValues(weights, closePoints, elementData, j, numElements);

      if (info.maxIdx1 != -1 && info.maxIdx2 != -1 && info.minIdx1 != -1 &&
          info.minIdx2 != -1) {
        weightsCopy[info.minIdx1] = NumericType(0);
        weightsCopy[info.minIdx2] = NumericType(0);
        weightsCopy[info.maxIdx1] = NumericType(0);
        weightsCopy[info.maxIdx2] = NumericType(0);
      }
    } else if (discard2 && numClosePoints > 2) {
      const auto info =
          findMinMaxValues(weights, closePoints, elementData, j, numElements);

      if (info.minIdx1 != -1 && info.maxIdx1 != -1) {
        weightsCopy[info.minIdx1] = NumericType(0);
        weightsCopy[info.maxIdx1] = NumericType(0);
      }
    }
  }

  template <class AT, class BT, std::size_t D>
  AT DotProductNT(const std::array<AT, D> &pVecA,
                  const std::array<BT, D> &pVecB) {
    AT dot = 0;
    for (size_t i = 0; i < D; ++i) {
      dot += pVecA[i] * pVecB[i];
    }
    return dot;
  }
};

} // namespace viennaps::gpu
