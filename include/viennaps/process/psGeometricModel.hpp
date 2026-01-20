#pragma once

#include <lsGeometricAdvect.hpp>

#include "../psDomain.hpp"

namespace viennaps {

using namespace viennacore;

VIENNAPS_TEMPLATE_ND(NumericType, D) class GeometricModel {
  SmartPointer<Domain<NumericType, D>> domain = nullptr;
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

  void setDomain(SmartPointer<Domain<NumericType, D>> passedDomain) {
    domain = passedDomain;
  }

  void setDistribution(
      SmartPointer<viennals::GeometricAdvectDistribution<NumericType, D>>
          passedDist) {
    dist = passedDist;
  }

  void setMask(SmartPointer<viennals::Domain<NumericType, D>> passedMask) {
    mask = passedMask;
  }

  void apply() {
    if (!dist) {
      VIENNACORE_LOG_ERROR(
          "No GeometricAdvectDistribution passed to GeometricModel.");
      return;
    }

    viennals::GeometricAdvect<NumericType, D>(domain->getLevelSets().back(),
                                              dist, mask)
        .apply();

    for (int i = domain->getNumberOfLevelSets() - 1; i >= 0; --i) {
      viennals::BooleanOperation<NumericType, D>(
          domain->getLevelSets()[i], domain->getLevelSets().back(),
          viennals::BooleanOperationEnum::INTERSECT)
          .apply();
    }
  }
};

} // namespace viennaps
