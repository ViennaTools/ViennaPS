#pragma once

#include <psSurfaceModel.hpp>

template <typename NumericType>
class SurfaceModel : public psSurfaceModel<NumericType> {
public:
  psSmartPointer<std::vector<NumericType>> calculateVelocities(
      psSmartPointer<psPointData<NumericType>> Rates,
      const std::vector<std::array<NumericType, 3>> &coordinates,
      const std::vector<NumericType> &materialIds) override {

    auto flux =
        Rates->getScalarData("particleRate") std::vector<NumericType> rates(
            materialIds.size(), 0.);
    for (std::size_t i = 0; i < rates.size(); i++) {
      if (!psMaterialMap::isMaterial(materialIds[i], psMaterial::Mask)) {
        rates[i] = flux->at(i);
      }
    }

    return psSmartPointer<std::vector<NumericType>>::New(rates);
  }
};