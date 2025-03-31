#pragma once

#include <lsMesh.hpp>

#include <raygIndexMap.hpp>
#include <raygParticle.hpp>

#include <vcCudaBuffer.hpp>
#include <vcKDTree.hpp>

#include <numeric>

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

    auto numData = indexMap_.getNumberOfData();
    const auto &points = diskMesh_->nodes;
    auto numPoints = points.size();
    auto numElements = elementKdTree_->getNumberOfPoints();
    auto normals = diskMesh_->cellData.getVectorData("Normals");
    auto elementNormals = surfaceMesh_->cellData.getVectorData("Normals");
    const float sqrdDist = conversionRadius_ * conversionRadius_;

    // retrieve data from device
    std::vector<MeshNT> elementData(numData * numElements);
    d_elementData_.download(elementData.data(), numData * numElements);

    // prepare point data container
    pointData_->clear();
    for (const auto &label : indexMap_) {
      std::vector<NumericType> data(numPoints, 0.);
      pointData_->insertNextScalarData(std::move(data), label);
    }
    assert(pointData_->getScalarDataSize() == numData); // assert number of data

#pragma omp parallel for
    for (unsigned i = 0; i < numPoints; i++) {

      auto closePoints =
          elementKdTree_->findNearestWithinRadius(points[i],
                                                  sqrdDist)
              .value(); // we have to use the squared distance here

      std::vector<NumericType> weights(closePoints.size(), 0.);

      unsigned numClosePoints = 0;
      for (int n = 0; n < closePoints.size(); n++) {
        const auto &p = closePoints[n];
        assert(p.first < elementNormals->size());

        auto weight = DotProductNT(normals->at(i), elementNormals->at(p.first));

        if (weight > 1e-6 && !std::isnan(weight)) {
          weights[n] = weight;
          numClosePoints++;
        }
      }

      std::size_t nearestIdx = 0;
      if (numClosePoints == 0) { // fallback to nearest point
        auto nearestPoint = elementKdTree_->findNearest(points[i]);
        nearestIdx = nearestPoint->first;
      }

      for (unsigned j = 0; j < numData; j++) {

        NumericType value = 0.;
        auto weightsCopy = weights;

        if (numClosePoints > 0) { // perform weighted average

          if (discard4 && numClosePoints > 4) { // Discard 2 min and max value
            NumericType min1, min2, max1, max2;
            min1 = min2 = std::numeric_limits<NumericType>::max();
            max1 = max2 = 0.;
            int minIdx1 = -1, minIdx2 = -1, maxIdx1 = -1, maxIdx2 = -1;

            for (int k = 0; k < closePoints.size(); k++) {
              if (weights[k] != 0.) {
                unsigned elementIdx = closePoints[k].first + j * numElements;
                if (elementData[elementIdx] < min1) {
                  min2 = min1;
                  minIdx2 = minIdx1;
                  min1 = elementData[elementIdx];
                  minIdx1 = k;
                } else if (elementData[elementIdx] < min2) {
                  min2 = elementData[elementIdx];
                  minIdx2 = k;
                }
                if (elementData[elementIdx] > max1) {
                  max2 = max1;
                  maxIdx2 = maxIdx1;
                  max1 = elementData[elementIdx];
                  maxIdx1 = k;
                } else if (elementData[elementIdx] > max2) {
                  max2 = elementData[elementIdx];
                  maxIdx2 = k;
                }
              }
            }

            if (!(maxIdx1 == -1 || maxIdx2 == -1 || minIdx1 == -1 ||
                  minIdx2 == -1)) {
              // discard values by setting their weight to 0
              weightsCopy[minIdx1] = 0.;
              weightsCopy[minIdx2] = 0.;
              weightsCopy[maxIdx1] = 0.;
              weightsCopy[maxIdx2] = 0.;
            }

          } else if (discard2 &&
                     numClosePoints > 2) { // Discard min and max value

            NumericType minValue = std::numeric_limits<NumericType>::max();
            NumericType maxValue = 0.;
            int minIdx = -1; // in weights vector
            int maxIdx = -1;

            for (int k = 0; k < closePoints.size(); k++) {
              if (weights[k] != 0.) {
                unsigned elementIdx = closePoints[k].first + j * numElements;
                if (elementData[elementIdx] < minValue) {
                  minValue = elementData[elementIdx];
                  minIdx = k;
                }
                if (elementData[elementIdx] > maxValue) {
                  maxValue = elementData[elementIdx];
                  maxIdx = k;
                }
              }
            }
            if (minIdx != -1 && maxIdx != -1) {
              // discard values by setting their weight to 0
              weightsCopy[minIdx] = 0.;
              weightsCopy[maxIdx] = 0.;
            }
          }

          NumericType sum =
              std::accumulate(weightsCopy.begin(), weightsCopy.end(), 0.);
          assert(sum != 0.);

          for (int k = 0; k < closePoints.size(); k++) {
            value += weightsCopy[k] *
                     elementData[closePoints[k].first + j * numElements];
          }

          value /= sum;

        } else {
          // fallback to nearest point
          value = elementData[nearestIdx + j * numElements];
        }

        pointData_->getScalarData(j)->at(i) = value;
      }
    }
  }

private:
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
