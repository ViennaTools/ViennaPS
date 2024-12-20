#pragma once

#include <curtIndexMap.hpp>
#include <curtParticle.hpp>
#include <gpu/vcCudaBuffer.hpp>
#include <lsMesh.hpp>
#include <vcKDTree.hpp>

namespace viennaps {

namespace gpu {

using namespace viennacore;

template <typename NumericType> class ElementToPointData {
  CudaBuffer &d_elementData_;
  SmartPointer<viennals::PointData<NumericType>> pointData_;
  const IndexMap indexMap_;
  SmartPointer<KDTree<float, Vec3Df>> elementKdTree_;
  SmartPointer<viennals::Mesh<NumericType>> diskMesh_;
  SmartPointer<viennals::Mesh<float>> surfaceMesh_;
  const NumericType conversionRadius_;

public:
  ElementToPointData(CudaBuffer &d_elementData,
                     SmartPointer<viennals::PointData<NumericType>> pointData,
                     const std::vector<Particle<NumericType>> &particles,
                     SmartPointer<KDTree<float, Vec3Df>> elementKdTree,
                     SmartPointer<viennals::Mesh<NumericType>> diskMesh,
                     SmartPointer<viennals::Mesh<float>> surfMesh,
                     const NumericType conversionRadius)
      : d_elementData_(d_elementData), pointData_(pointData),
        indexMap_(particles), elementKdTree_(elementKdTree),
        diskMesh_(diskMesh), surfaceMesh_(surfMesh),
        conversionRadius_(conversionRadius) {}

  ElementToPointData(CudaBuffer &d_elementData,
                     SmartPointer<viennals::PointData<NumericType>> pointData,
                     const IndexMap &indexMap,
                     SmartPointer<KDTree<float, Vec3Df>> elementKdTree,
                     SmartPointer<viennals::Mesh<NumericType>> diskMesh,
                     SmartPointer<viennals::Mesh<float>> surfMesh,
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
    std::vector<NumericType> elementData(numData * numElements);
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

      auto closePoints = elementKdTree_->findNearestWithinRadius(
          points[i], sqrdDist); // we have to use the squared distance here

      std::vector<NumericType> weights;
      weights.reserve(closePoints.value().size());

      unsigned numClosePoints = 0;
      for (auto p : closePoints.value()) {
        assert(p.first < elementNormals->size());

        auto weight = DotProduct(normals->at(i), elementNormals->at(p.first));

        if (weight > 1e-6 && !std::isnan(weight)) {
          weights.push_back(weight);
          numClosePoints++;
        } else {
          weights.push_back(0.);
        }
      }
      assert(weights.size() == closePoints.value().size());

      std::size_t nearestIdx = 0;
      if (numClosePoints == 0) { // fallback to nearest point
        auto nearestPoint = elementKdTree_->findNearest(points[i]);
        nearestIdx = nearestPoint->first;
      }

      for (unsigned j = 0; j < numData; j++) {

        NumericType value = 0.;
        auto weightsCopy = weights;

        if (numClosePoints > 0) { // perform weighted average

          // Discard min and max
          if (numClosePoints > 2) {
            NumericType minValue = std::numeric_limits<NumericType>::max();
            NumericType maxValue = 0.;
            unsigned minIdx = 0; // in weights vector
            unsigned maxIdx = 0;

            unsigned k = 0;
            for (auto p : closePoints.value()) {
              if (weights[k] > 1e-6 && !std::isnan(weights[k])) {
                unsigned elementIdx = p.first + j * numElements;
                if (elementData[elementIdx] < minValue) {
                  minValue = elementData[elementIdx];
                  minIdx = k;
                }
                if (elementData[elementIdx] > maxValue) {
                  maxValue = elementData[elementIdx];
                  maxIdx = k;
                }
              }
              k++;
            }

            // discard values by setting their weight to 0
            weightsCopy[minIdx] = 0.;
            weightsCopy[maxIdx] = 0.;
          }

          // calculate weight sum
          NumericType sum =
              std::accumulate(weightsCopy.begin(), weightsCopy.end(), 0.);
          assert(sum != 0.);

          unsigned n = 0;
          for (auto p : closePoints.value()) {
            value += weightsCopy[n] * elementData[p.first + j * numElements];
            n++;
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
};

} // namespace gpu
} // namespace viennaps
