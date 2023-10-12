#pragma once

#include <psSurfaceModel.hpp>

template <typename NumericType>
class SurfaceModel : public psSurfaceModel<NumericType> {
public:
  using psSurfaceModel<NumericType>::Coverages;
  using psSurfaceModel<NumericType>::processParams;

  void initializeCoverages(unsigned numGeometryPoints) override {
    std::vector<NumericType> someCoverages(numGeometryPoints, 0);

    Coverages = psSmartPointer<psPointData<NumericType>>::New();
    Coverages->insertNextScalarData(someCoverages, "coverages");
  }

  void initializeProcessParameters() override {
    processParams = psSmartPointer<psProcessParams<NumericType>>::New();
    processParams->insertNextScalar(0., "processParameter");
  }

  psSmartPointer<std::vector<NumericType>> calculateVelocities(
      psSmartPointer<psPointData<NumericType>> Rates,
      const std::vector<std::array<NumericType, 3>> &coordinates,
      const std::vector<NumericType> &materialIds) override {
    // use coverages and rates here to calculate the velocity here
    return psSmartPointer<std::vector<NumericType>>::New(
        *Rates->getScalarData("particleRate"));
  }

  void
  updateCoverages(psSmartPointer<psPointData<NumericType>> Rates) override {
    // update coverages
  }
};