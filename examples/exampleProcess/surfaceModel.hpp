#pragma once

#include <psSurfaceModel.hpp>

template <typename NumericType>
class SurfaceModel : public psSurfaceModel<NumericType> {
public:
  using psSurfaceModel<NumericType>::coverages;
  using psSurfaceModel<NumericType>::processParams;

  void initializeCoverages(unsigned numGeometryPoints) override {
    std::vector<NumericType> someCoverages(numGeometryPoints, 0);

    coverages = psSmartPointer<psPointData<NumericType>>::New();
    coverages->insertNextScalarData(someCoverages, "coverages");
  }

  void initializeProcessParameters() override {
    processParams = psSmartPointer<psProcessParams<NumericType>>::New();
    processParams->insertNextScalar(0., "processParameter");
  }

  psSmartPointer<std::vector<NumericType>> calculateVelocities(
      psSmartPointer<psPointData<NumericType>> rates,
      const std::vector<std::array<NumericType, 3>> &coordinates,
      const std::vector<NumericType> &materialIds) override {
    // use coverages and rates here to calculate the velocity here
    return psSmartPointer<std::vector<NumericType>>::New(
        *rates->getScalarData("particleRate"));
  }

  void updateCoverages(psSmartPointer<psPointData<NumericType>> rates,
                       const std::vector<NumericType> &materialIds) override {
    // update coverages
  }
};