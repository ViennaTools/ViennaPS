#pragma once

#include <lsMesh.hpp>
#include <rayParticle.hpp>
#include <vcKDTree.hpp>

#include <algorithm>
#include <cassert>
#include <numeric>
#include <utility>

namespace viennaps {

using namespace viennacore;

class IndexMap {
  std::vector<std::string> dataLabels;

public:
  IndexMap() = default;

#ifdef VIENNACORE_COMPILE_GPU
  template <class T>
  explicit IndexMap(const std::vector<viennaray::gpu::Particle<T>> &particles) {
    for (size_t pIdx = 0; pIdx < particles.size(); pIdx++) {
      for (size_t dIdx = 0; dIdx < particles[pIdx].dataLabels.size(); dIdx++) {
        dataLabels.push_back(particles[pIdx].dataLabels[dIdx]);
      }
    }
  }
#endif

  template <class T>
  explicit IndexMap(
      const std::vector<std::unique_ptr<viennaray::AbstractParticle<T>>>
          &particles) {
    for (size_t pIdx = 0; pIdx < particles.size(); pIdx++) {
      auto labels = particles[pIdx]->getLocalDataLabels();
      for (size_t dIdx = 0; dIdx < labels.size(); dIdx++) {
        dataLabels.push_back(labels[dIdx]);
      }
    }
  }

  void insertNextDataLabel(std::string dataLabel) {
    dataLabels.push_back(std::move(dataLabel));
  }

  std::size_t getIndex(const std::string &label) const {
    for (std::size_t idx = 0; idx < dataLabels.size(); idx++) {
      if (dataLabels[idx] == label) {
        return idx;
      }
    }
    assert(false && "Data label not found");
    return 0;
  }

  [[nodiscard]] const std::string &getLabel(std::size_t idx) const {
    assert(idx < dataLabels.size());
    return dataLabels[idx];
  }

  std::size_t getNumberOfData() const { return dataLabels.size(); }

  std::vector<std::string>::const_iterator begin() const {
    return dataLabels.cbegin();
  }

  std::vector<std::string>::const_iterator end() const {
    return dataLabels.cend();
  }
};

template <typename NumericType, typename MeshNT, typename ResultType,
          bool d2 = true, bool d4 = true>
class ElementToPointData {
  const IndexMap indexMap_;
  SmartPointer<viennals::PointData<NumericType>> pointData_;
  SmartPointer<KDTree<NumericType, Vec3D<NumericType>>> elementKdTree_;
  SmartPointer<viennals::Mesh<NumericType>> diskMesh_;
  SmartPointer<viennals::Mesh<MeshNT>> surfaceMesh_;
  const NumericType conversionRadius_;

  std::vector<std::vector<ResultType>> elementDataArrays_;
  std::vector<std::tuple<std::vector<size_t>, std::vector<double>, unsigned>>
      closeElements_;

  static constexpr bool discard2 = d2;
  static constexpr bool discard4 = d4;

public:
  ElementToPointData(
      IndexMap indexMap,
      SmartPointer<viennals::PointData<NumericType>>
          pointData, // target point data
      SmartPointer<KDTree<NumericType, Vec3D<NumericType>>> elementKdTree,
      SmartPointer<viennals::Mesh<NumericType>> diskMesh,
      SmartPointer<viennals::Mesh<MeshNT>> surfMesh,
      const NumericType conversionRadius)
      : indexMap_(std::move(indexMap)), pointData_(pointData),
        elementKdTree_(elementKdTree), diskMesh_(diskMesh),
        surfaceMesh_(surfMesh), conversionRadius_(conversionRadius) {}

  void setElementDataArrays(
      std::vector<std::vector<ResultType>> &&elementDataArrays) {
    elementDataArrays_ = std::move(elementDataArrays);
  }

  void prepare() {
    const auto numData = indexMap_.getNumberOfData();
    const auto &points = diskMesh_->nodes;
    const auto numPoints = points.size();
    const auto numElements = elementKdTree_->getNumberOfPoints();
    const auto normals = diskMesh_->cellData.getVectorData("Normals");
    const auto elementNormals = surfaceMesh_->cellData.getVectorData("Normals");

    // prepare point data container
    pointData_->clear();
    for (const auto &label : indexMap_) {
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

      assert(!weights.empty() && !elementIndices.empty());
      closeElements_[i] =
          std::make_tuple(elementIndices, weights, numClosePoints);
    }
  }

  void convert() {
    const auto numData = indexMap_.getNumberOfData();
    const auto &points = diskMesh_->nodes;
    const auto numPoints = points.size();

    if (elementDataArrays_.size() != numData) {
      VIENNACORE_LOG_ERROR(
          "ElementToPointData: "
          "Number of data arrays does not match expected count.");
    }

#pragma omp parallel for schedule(static)
    for (unsigned i = 0; i < numPoints; i++) {

      const auto &[closeElements, weights, numClosePoints] = closeElements_[i];

      for (unsigned j = 0; j < numData; ++j) {
        NumericType value = NumericType(0);
        const auto &elementData = elementDataArrays_[j];
        auto pointData = pointData_->getScalarData(j);

        // Discard outlier values if enabled
        const auto weightsCopy = discardOutliers(weights, closeElements,
                                                 elementData, numClosePoints);

        // Compute weighted average
        const double sum =
            std::accumulate(weightsCopy.cbegin(), weightsCopy.cend(), 0.0);

        if (sum > 1e-6) {
          for (size_t k = 0; k < closeElements.size(); ++k) {
            if (weightsCopy[k] > 0.0) {
              value += weightsCopy[k] * elementData[closeElements[k]];
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
