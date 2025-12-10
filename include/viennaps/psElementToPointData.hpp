#pragma once

#include <lsMesh.hpp>

#include <rayParticle.hpp>

#include <utility>
#include <vcCudaBuffer.hpp>
#include <vcKDTree.hpp>

#include <algorithm>
#include <cassert>
#include <numeric>

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
class ElementToPointDataBase {
protected:
  const IndexMap indexMap_;
  SmartPointer<viennals::PointData<NumericType>> pointData_;
  SmartPointer<KDTree<NumericType, Vec3D<NumericType>>> elementKdTree_;
  SmartPointer<viennals::Mesh<NumericType>> diskMesh_;
  SmartPointer<viennals::Mesh<MeshNT>> surfaceMesh_;
  const NumericType conversionRadius_;

  static constexpr bool discard2 = d2;
  static constexpr bool discard4 = d4;

public:
  virtual ~ElementToPointDataBase() = default;
  ElementToPointDataBase(
      IndexMap indexMap,
      SmartPointer<viennals::PointData<NumericType>> pointData,
      SmartPointer<KDTree<NumericType, Vec3D<NumericType>>> elementKdTree,
      SmartPointer<viennals::Mesh<NumericType>> diskMesh,
      SmartPointer<viennals::Mesh<MeshNT>> surfMesh,
      const NumericType conversionRadius)
      : indexMap_(std::move(indexMap)), pointData_(pointData),
        elementKdTree_(elementKdTree), diskMesh_(diskMesh),
        surfaceMesh_(surfMesh), conversionRadius_(conversionRadius) {}

  void apply() {
    const auto numData = indexMap_.getNumberOfData();
    const auto &points = diskMesh_->nodes;
    const auto numPoints = points.size();
    const auto numElements = elementKdTree_->getNumberOfPoints();
    const auto normals = diskMesh_->cellData.getVectorData("Normals");
    const auto elementNormals = surfaceMesh_->cellData.getVectorData("Normals");

    // retrieve data from device
    std::vector<ResultType> elementData;
    flattenElementData(elementData);
    assert(elementData.size() == numData * numElements);

    // prepare point data container
    pointData_->clear();
    for (const auto &label : indexMap_) {
      std::vector<NumericType> data(numPoints, 0.);
      pointData_->insertNextScalarData(std::move(data), label);
    }

    if (pointData_->getScalarDataSize() != numData) {
      VIENNACORE_LOG_ERROR(
          "ElementToPointData: "
          "Number of data arrays does not match expected count.");
    }

#pragma omp parallel for schedule(static)
    for (unsigned i = 0; i < numPoints; i++) {

      // we have to use the squared distance here
      auto closeElements =
          elementKdTree_->findNearestWithinRadius(points[i], conversionRadius_)
              .value();

      std::vector<double> weights(closeElements.size(), 0.);

      // compute weights based on normal alignment
      unsigned numClosePoints = 0;
      for (std::size_t n = 0; n < closeElements.size(); ++n) {
        const auto &p = closeElements[n];
        assert(p.first < numElements);

        const double weight =
            DotProductNT(normals->at(i), elementNormals->at(p.first));

        if (weight > 1e-6 && !std::isnan(weight)) {
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

          // Discard outlier values if enabled
          const auto weightsCopy =
              discardOutliers(weights, closeElements, elementData, j,
                              numElements, numClosePoints);

          // Compute weighted average
          const double sum =
              std::accumulate(weightsCopy.cbegin(), weightsCopy.cend(), 0.0);

          if (sum > 1e-6) {
            for (std::size_t k = 0; k < closeElements.size(); ++k) {
              if (weightsCopy[k] > 0.0) {
                const unsigned flattIdx =
                    closeElements[k].first + j * numElements;
                value += weightsCopy[k] * elementData[flattIdx];
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
          assert(numClosePoints == 0);
          value = elementData[nearestIdx + j * numElements];
        }

        pointData_->getScalarData(j)->at(i) = value;
      }
    }
  }

protected:
  virtual void
  flattenElementData(std::vector<ResultType> &elementData) const = 0;

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

  static MinMaxInfo findMinMaxValues(
      const std::vector<double> &weights,
      const std::vector<std::pair<std::size_t, NumericType>> &closePoints,
      const std::vector<ResultType> &elementData, unsigned j,
      unsigned numElements) {
    MinMaxInfo info;

    for (std::size_t k = 0; k < closePoints.size(); ++k) {
      if (weights[k] > 0.0) {
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

  static auto discardOutliers(
      const std::vector<double> &weights,
      const std::vector<std::pair<std::size_t, NumericType>> &closePoints,
      const std::vector<ResultType> &elementData, unsigned j,
      unsigned numElements, unsigned numClosePoints) {

    // copy weights to modify
    auto weightsCopy = weights;

    if (discard4 && numClosePoints > 4) {
      const auto info =
          findMinMaxValues(weights, closePoints, elementData, j, numElements);

      if (info.maxIdx1 != -1 && info.maxIdx2 != -1 && info.minIdx1 != -1 &&
          info.minIdx2 != -1) {
        weightsCopy[info.minIdx1] = 0.0;
        weightsCopy[info.minIdx2] = 0.0;
        weightsCopy[info.maxIdx1] = 0.0;
        weightsCopy[info.maxIdx2] = 0.0;
      }
    } else if (discard2 && numClosePoints > 2) {
      const auto info =
          findMinMaxValues(weights, closePoints, elementData, j, numElements);

      if (info.minIdx1 != -1 && info.maxIdx1 != -1) {
        weightsCopy[info.minIdx1] = 0.0;
        weightsCopy[info.maxIdx1] = 0.0;
      }
    }

    return weightsCopy;
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

template <typename NumericType, typename MeshNT, typename ResultType,
          bool d2 = true, bool d4 = true>
class ElementToPointData
    : public ElementToPointDataBase<NumericType, MeshNT, ResultType, d2, d4> {
public:
  ElementToPointData(
      const std::vector<std::vector<NumericType>> &elementData,
      SmartPointer<viennals::PointData<NumericType>> pointData,
      const IndexMap &indexMap,
      SmartPointer<KDTree<NumericType, Vec3D<NumericType>>> elementKdTree,
      SmartPointer<viennals::Mesh<NumericType>> diskMesh,
      SmartPointer<viennals::Mesh<MeshNT>> surfMesh,
      const NumericType conversionRadius)
      : ::viennaps::ElementToPointDataBase<NumericType, MeshNT, ResultType, d2,
                                           d4>(indexMap, pointData,
                                               elementKdTree, diskMesh,
                                               surfMesh, conversionRadius),
        elementData_(elementData) {}

  ElementToPointData(
      const std::vector<std::vector<NumericType>> &elementData,
      SmartPointer<viennals::PointData<NumericType>> pointData,
      const std::vector<
          std::unique_ptr<viennaray::AbstractParticle<NumericType>>> &particles,
      SmartPointer<KDTree<NumericType, Vec3D<NumericType>>> elementKdTree,
      SmartPointer<viennals::Mesh<NumericType>> diskMesh,
      SmartPointer<viennals::Mesh<MeshNT>> surfMesh,
      const NumericType conversionRadius)
      : viennaps::ElementToPointDataBase<NumericType, MeshNT, ResultType, d2,
                                         d4>(viennaps::IndexMap(particles),
                                             pointData, elementKdTree, diskMesh,
                                             surfMesh, conversionRadius),
        elementData_(elementData) {}

protected:
  void flattenElementData(std::vector<ResultType> &elementData) const override {
    const auto numData = elementData_.size();
    assert(elementData_.size() > 0);
    const auto numElements = elementData_[0].size();

    elementData.resize(numData * numElements);
#pragma omp parallel for schedule(static)
    for (unsigned i = 0; i < numData; ++i) {
      const unsigned offset = i * numElements;
      for (unsigned j = 0; j < numElements; ++j) {
        elementData[offset + j] = elementData_[i][j];
      }
    }
  }

private:
  const std::vector<std::vector<NumericType>> &elementData_;
};

#ifdef VIENNACORE_COMPILE_GPU
namespace gpu {
template <typename NumericType, typename MeshNT, typename ResultType,
          bool d2 = true, bool d4 = true>
class ElementToPointData
    : public ElementToPointDataBase<NumericType, MeshNT, ResultType, d2, d4> {
public:
  ElementToPointData(
      CudaBuffer &d_elementData,
      SmartPointer<viennals::PointData<NumericType>> pointData,
      const IndexMap &indexMap,
      SmartPointer<KDTree<NumericType, Vec3D<NumericType>>> elementKdTree,
      SmartPointer<viennals::Mesh<NumericType>> diskMesh,
      SmartPointer<viennals::Mesh<MeshNT>> surfMesh,
      const NumericType conversionRadius)
      : ::viennaps::ElementToPointDataBase<NumericType, MeshNT, ResultType, d2,
                                           d4>(indexMap, pointData,
                                               elementKdTree, diskMesh,
                                               surfMesh, conversionRadius),
        d_elementData_(d_elementData) {}

  ElementToPointData(
      CudaBuffer &d_elementData,
      SmartPointer<viennals::PointData<NumericType>> pointData,
      const std::vector<viennaray::gpu::Particle<NumericType>> &particles,
      SmartPointer<KDTree<NumericType, Vec3D<NumericType>>> elementKdTree,
      SmartPointer<viennals::Mesh<NumericType>> diskMesh,
      SmartPointer<viennals::Mesh<MeshNT>> surfMesh,
      const NumericType conversionRadius)
      : viennaps::ElementToPointDataBase<NumericType, MeshNT, ResultType, d2,
                                         d4>(viennaps::IndexMap(particles),
                                             pointData, elementKdTree, diskMesh,
                                             surfMesh, conversionRadius),
        d_elementData_(d_elementData) {}

protected:
  void flattenElementData(std::vector<ResultType> &elementData) const override {
    const auto numData = this->indexMap_.getNumberOfData();
    const auto numElements = this->elementKdTree_->getNumberOfPoints();

    // retrieve data from device
    elementData.resize(numData * numElements);
    d_elementData_.download(elementData.data(), numData * numElements);
  }

private:
  CudaBuffer &d_elementData_;
};
} // namespace gpu
#endif

} // namespace viennaps
