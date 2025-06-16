#pragma once

#include <psSurfaceModel.hpp>

template <typename NumericType>
class SurfaceModel : public viennaps::SurfaceModel<NumericType> {
public:
  using viennaps::SurfaceModel<NumericType>::coverages;
  using viennaps::SurfaceModel<NumericType>::processParams;

  void initializeCoverages(unsigned numGeometryPoints) override {
    std::vector<NumericType> someCoverages(numGeometryPoints, 0);

    coverages = viennals::PointData<NumericType>::New();
    coverages->insertNextScalarData(someCoverages, "coverages");
  }

  void initializeProcessParameters() override {
    processParams =
        viennaps::SmartPointer<viennaps::ProcessParams<NumericType>>::New();
    processParams->insertNextScalar(0., "processParameter");
  }

  viennaps::SmartPointer<std::vector<NumericType>> calculateVelocities(
      viennaps::SmartPointer<viennals::PointData<NumericType>> rates,
      const std::vector<viennaps::Vec3D<NumericType>> &coordinates,
      const std::vector<NumericType> &materialIds) override {
    // use coverages and rates here to calculate the velocity here
    return viennaps::SmartPointer<std::vector<NumericType>>::New(
        *rates->getScalarData("particleRate"));
  }

  void updateCoverages(
      viennaps::SmartPointer<viennals::PointData<NumericType>> rates,
      const std::vector<NumericType> &materialIds) override {
    // update coverages
  }
};