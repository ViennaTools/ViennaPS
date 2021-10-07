#pragma once

#include <psSurfaceModel.hpp>

template <typename NumericType>
class surfaceModel : public psSurfaceModel<NumericType> {
public:
  void initializeCoverages(unsigned numGeometryPoints) override {}

  psSmartPointer<std::vector<NumericType>>
  calculateVelocities(psSmartPointer<psPointData<NumericType>> Rates,
                      const std::vector<NumericType> &materialIds,
                      const long numRaysPerPoint) override {
    std::vector<NumericType> velocities(materialIds.size(), 0.);

    auto rate = Rates->getScalarData("particleRate");

    // normalize rate
    double max = 0;
    for (size_t i = 0; i < velocities.size(); ++i) {
      if (rate->at(i) > max)
        max = rate->at(i);
    }
    for (size_t i = 0; i < velocities.size(); ++i) {
      velocities[i] = rate->at(i) / max;
    }

    return psSmartPointer<std::vector<NumericType>>::New(velocities);
  }

  void updateCoverages(psSmartPointer<psPointData<NumericType>> Rates,
                       const long numRaysPerPoint) override {}
};