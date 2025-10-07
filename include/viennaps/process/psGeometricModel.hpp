#pragma once

#include <lsGeometricAdvect.hpp>

#include "../psDomain.hpp"

namespace viennaps {

using namespace viennacore;

template <typename NumericType, int D> class GeometricModel {
  SmartPointer<Domain<NumericType, D>> domain = nullptr;
  SmartPointer<viennals::GeometricAdvectDistribution<viennahrle::CoordType, D>>
      dist = nullptr;
  SmartPointer<viennals::Domain<NumericType, D>> mask = nullptr;

public:
  GeometricModel() = default;

  GeometricModel(
      SmartPointer<
          viennals::GeometricAdvectDistribution<viennahrle::CoordType, D>>
          passedDist,
      SmartPointer<viennals::Domain<NumericType, D>> passedMask = nullptr)
      : dist(passedDist), mask(passedMask) {}

  void setDomain(SmartPointer<Domain<NumericType, D>> passedDomain) {
    domain = passedDomain;
  }

  void setDistribution(
      SmartPointer<
          viennals::GeometricAdvectDistribution<viennahrle::CoordType, D>>
          passedDist) {
    dist = passedDist;
  }

  void setMask(SmartPointer<viennals::Domain<NumericType, D>> passedMask) {
    mask = passedMask;
  }

  void apply() {
    if (!dist) {
      Logger::getInstance()
          .addError("No GeometricAdvectDistribution passed to GeometricModel.")
          .print();
    }

    viennals::GeometricAdvect<NumericType, D>(domain->getLevelSets().back(),
                                              dist, mask)
        .apply();
  }
};

} // namespace viennaps
