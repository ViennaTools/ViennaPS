#pragma once

#include "psPreCompileMacros.hpp"

#include <lsMesh.hpp>
#include <rayParticle.hpp>
#include <vcKDTree.hpp>

#include <algorithm>
#include <cassert>
#include <numeric>
#include <utility>

namespace viennaps {

using namespace viennacore;

template <class classType, class MeshNT, class ResultType, bool d2 = true,
          bool d4 = true>
class ElementToPointData {
  const std::vector<std::string> dataLabels_;
  SmartPointer<viennals::PointData<NumericType>> pointData_;
  SmartPointer<KDTree<NumericType, Vec3D<NumericType>>> elementKdTree_;
  SmartPointer<viennals::Mesh<NumericType>> diskMesh_;
  SmartPointer<viennals::Mesh<MeshNT>> surfaceMesh_;
  const NumericType conversionRadius_;

  struct CloseElements {
    std::vector<size_t> indices;
    std::vector<double> weights;
    unsigned numClosePoints;
  };

  std::vector<std::vector<ResultType>> elementDataArrays_;
  std::vector<CloseElements> closeElements_;
  static constexpr bool discard2 = d2;
  static constexpr bool discard4 = d4;

public:
  ElementToPointData(
      const std::vector<std::string> &dataLabels,
      SmartPointer<viennals::PointData<NumericType>>
          pointData, // target point data
      SmartPointer<KDTree<NumericType, Vec3D<NumericType>>> elementKdTree,
      SmartPointer<viennals::Mesh<NumericType>> diskMesh,
      SmartPointer<viennals::Mesh<MeshNT>> surfMesh,
      const NumericType conversionRadius)
      : dataLabels_(dataLabels), pointData_(pointData),
        elementKdTree_(elementKdTree), diskMesh_(diskMesh),
        surfaceMesh_(surfMesh), conversionRadius_(conversionRadius) {}

  void setElementDataArrays(
      std::vector<std::vector<ResultType>> &&elementDataArrays) {
    elementDataArrays_ = std::move(elementDataArrays);
  }

  void prepare() {
    const auto numData = dataLabels_.size();
    const auto &points = diskMesh_->nodes;
    const auto numPoints = points.size();
    const auto numElements = elementKdTree_->getNumberOfPoints();
    const auto normals = diskMesh_->cellData.getVectorData("Normals");
    const auto elementNormals = surfaceMesh_->cellData.getVectorData("Normals");

    // prepare point data container
    pointData_->clear();
    for (const auto &label : dataLabels_) {
      std::vector<NumericType> data(numPoints, 0.);
      pointData_->insertNextScalarData(std::move(data), label);
    }

    closeElements_.resize(numPoints);

#pragma omp parallel for schedule(static)
    for (unsigned i = 0; i < numPoints; i++) {

      // we have to use the squared distance here
      auto closeElements =
          elementKdTree_->findNearestWithinRadius(points[i], conversionRadius_)
              .value();

      std::vector<double> weights(closeElements.size(), 0.);
      std::vector<size_t> elementIndices(closeElements.size(), 0);

      // compute weights based on normal alignment
      unsigned numClosePoints = 0;
      for (std::size_t n = 0; n < closeElements.size(); ++n) {
        const auto &p = closeElements[n];
        assert(p.first < numElements);

        const double weight =
            DotProductNT(normals->at(i), elementNormals->at(p.first));

        if (weight > 1e-6 && !std::isnan(weight)) {
          weights[n] = weight;
          elementIndices[n] = p.first;
          ++numClosePoints;
        }
      }

      assert(!weights.empty() && !elementIndices.empty() &&
             "Weights and element indices should not be empty here.");
      closeElements_[i].indices = std::move(elementIndices);
      closeElements_[i].weights = std::move(weights);
      closeElements_[i].numClosePoints = numClosePoints;
    }
  }

  void convert() {
    const auto numData = dataLabels_.size();
    const auto &points = diskMesh_->nodes;
    const auto numPoints = points.size();

    if (elementDataArrays_.size() != numData) {
      VIENNACORE_LOG_ERROR(
          "ElementToPointData: "
          "Number of data arrays does not match expected count.");
    }

#pragma omp parallel for schedule(static)
    for (unsigned i = 0; i < numPoints; i++) {

      const auto &[indices, weights, numClosePoints] = closeElements_[i];

      for (unsigned j = 0; j < numData; ++j) {
        NumericType value = NumericType(0);
        const auto &elementData = elementDataArrays_[j];
        auto pointData = pointData_->getScalarData(j);

        // Discard outlier values if enabled
        const auto weightsCopy =
            discardOutliers(weights, indices, elementData, numClosePoints);

        // Compute weighted average
        const double sum =
            std::accumulate(weightsCopy.cbegin(), weightsCopy.cend(), 0.0);

        if (sum > 1e-6) [[likely]] {
          for (size_t k = 0; k < indices.size(); ++k) {
            if (weightsCopy[k] > 0.0) {
              value += weightsCopy[k] * elementData[indices[k]];
            }
          }
          value /= sum;
        } else {
          // Fallback if all weights were discarded
          auto nearestPoint = elementKdTree_->findNearest(points[i]);
          value = elementData[nearestPoint->first];
        }

        pointData->at(i) = value;
      }
    }
  }

  void apply() {
    prepare();
    convert();
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

  static MinMaxInfo
  findMinMaxValues(const std::vector<double> &weights,
                   const std::vector<size_t> &closePoints,
                   const std::vector<ResultType> &elementData) {
    MinMaxInfo info;

    for (std::size_t k = 0; k < closePoints.size(); ++k) {
      if (weights[k] > 0.0) {
        const auto value = elementData[closePoints[k]];

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

  static auto discardOutliers(const std::vector<double> &weights,
                              const std::vector<size_t> &closePoints,
                              const std::vector<ResultType> &elementData,
                              unsigned numClosePoints) {

    // copy weights to modify
    auto weightsCopy = weights;

    if (discard4 && numClosePoints > 4) {
      const auto info = findMinMaxValues(weights, closePoints, elementData);

      if (info.maxIdx1 != -1 && info.maxIdx2 != -1 && info.minIdx1 != -1 &&
          info.minIdx2 != -1) {
        weightsCopy[info.minIdx1] = 0.0;
        weightsCopy[info.minIdx2] = 0.0;
        weightsCopy[info.maxIdx1] = 0.0;
        weightsCopy[info.maxIdx2] = 0.0;
      }
    } else if (discard2 && numClosePoints > 2) {
      const auto info = findMinMaxValues(weights, closePoints, elementData);

      if (info.minIdx1 != -1 && info.maxIdx1 != -1) {
        weightsCopy[info.minIdx1] = 0.0;
        weightsCopy[info.maxIdx1] = 0.0;
      }
    }

    return weightsCopy;
  }

  template <class AT, class BT>
  static double DotProductNT(const std::array<AT, 3> &pVecA,
                             const std::array<BT, 3> &pVecB) {
    double dot = 0.0;
    for (size_t i = 0; i < 3; ++i)
      dot += pVecA[i] * pVecB[i];
    return dot;
  }
};

} // namespace viennaps
