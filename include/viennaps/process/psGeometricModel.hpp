#pragma once

#include <lsGeometricAdvect.hpp>

#include "../psDomain.hpp"

namespace viennaps {

using namespace viennacore;

VIENNAPS_TEMPLATE_ND(NumericType, D) class GeometricModel {
  SmartPointer<viennals::GeometricAdvectDistribution<NumericType, D>> dist =
      nullptr;
  SmartPointer<viennals::Domain<NumericType, D>> mask = nullptr;

public:
  GeometricModel() = default;

  GeometricModel(
      SmartPointer<viennals::GeometricAdvectDistribution<NumericType, D>>
          passedDist,
      SmartPointer<viennals::Domain<NumericType, D>> passedMask = nullptr)
      : dist(passedDist), mask(passedMask) {}

  void setDistribution(
      SmartPointer<viennals::GeometricAdvectDistribution<NumericType, D>>
          passedDist) {
    dist = passedDist;
  }

  void setMask(SmartPointer<viennals::Domain<NumericType, D>> passedMask) {
    mask = passedMask;
  }

  auto &getDistribution() const { return dist; }
  auto &getMask() const { return mask; }
};

} // namespace viennaps
