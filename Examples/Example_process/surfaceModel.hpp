#pragma once

#include <psSurfaceModel.hpp>

template <typename NumericType>
class surfaceModel : public psSurfaceModel<NumericType> {
public:
  using psSurfaceModel<NumericType>::processParams;

  void initializeCoverages(unsigned numGeometryPoints) override {}

  void initializeProcessParameters() override {
    processParams = psSmartPointer<psProcessParams<NumericType>>::New();
    processParams->insertNextScalar(0., "processParameter");
  }

  psSmartPointer<std::vector<NumericType>>
  calculateVelocities(psSmartPointer<psPointData<NumericType>> Rates,
                      const std::vector<NumericType> &materialIds) override {
    auto rate = Rates->getScalarData("particleRate");

    return rate;
  }

  void
  updateCoverages(psSmartPointer<psPointData<NumericType>> Rates) override {}
};