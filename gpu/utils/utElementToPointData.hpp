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
  CudaBuffer d_elementData;
  SmartPointer<viennals::PointData<NumericType>> pointData;
  const IndexMap indexMap;
  SmartPointer<KDTree<float, Vec3Df>> elementKdTree;
  SmartPointer<viennals::Mesh<NumericType>> pointMesh;
  SmartPointer<viennals::Mesh<float>> surfMesh;
  const NumericType gridDelta;
  const NumericType smoothFlux_ = 1.;

public:
  ElementToPointData(CudaBuffer d_elementData,
                     SmartPointer<viennals::PointData<NumericType>> pointData,
                     const std::vector<Particle<NumericType>> &particles,
                     SmartPointer<KDTree<float, Vec3Df>> elementKdTree,
                     SmartPointer<viennals::Mesh<NumericType>> pointMesh,
                     SmartPointer<viennals::Mesh<float>> surfMesh,
                     const NumericType gridDelta,
                     const NumericType smoothFlux = 1.)
      : d_elementData(d_elementData), pointData(pointData), indexMap(particles),
        elementKdTree(elementKdTree), pointMesh(pointMesh), surfMesh(surfMesh),
        gridDelta(gridDelta), smoothFlux_(smoothFlux) {}

  void apply() {

    auto numData = indexMap.getNumberOfData();
    const auto &points = pointMesh->nodes;
    auto numPoints = points.size();
    auto numElements = elementKdTree->getNumberOfPoints();
    auto normals = pointMesh->cellData.getVectorData("Normals");
    auto elementNormals = surfMesh->cellData.getVectorData("Normals");
    const float sqrdDist = smoothFlux_ * smoothFlux_ * gridDelta * gridDelta;

    // retrieve data from device
    std::vector<NumericType> elementData(numData * numElements);
    d_elementData.download(elementData.data(), numData * numElements);

    // prepare point data container
    pointData->clear();
    for (const auto &label : indexMap) {
      std::vector<NumericType> data(numPoints, 0.);
      pointData->insertNextScalarData(std::move(data), label);
    }
    assert(pointData->getScalarDataSize() == numData); // assert number of data

#pragma omp parallel for
    for (unsigned i = 0; i < numPoints; i++) {

      auto closePoints = elementKdTree->findNearestWithinRadius(
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
        auto nearestPoint = elementKdTree->findNearest(points[i]);
        nearestIdx = nearestPoint->first;
      }

      for (unsigned j = 0; j < numData; j++) {

        NumericType value = 0.;

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

            weights[minIdx] = 0.;
            weights[maxIdx] = 0.;
          }

          // calculate weight sum
          NumericType sum = std::accumulate(weights.begin(), weights.end(), 0.);

          unsigned n = 0;
          for (auto p : closePoints.value()) {
            if (weights[n] > 0.)
              value += weights[n] * elementData[p.first + j * numElements];
            n++;
          }
          value /= sum;

        } else {
          // fallback to nearest point
          value = elementData[nearestIdx + j * numElements];
        }

        pointData->getScalarData(j)->at(i) = value;
      }
    }
  }
};

} // namespace gpu
} // namespace viennaps
