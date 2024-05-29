#pragma once

#include <type_traits>

#include <lsGeometricAdvect.hpp>
#include <lsGeometricAdvectDistributions.hpp>

#include "../psGeometricModel.hpp"
#include "../psProcessModel.hpp"

namespace viennaps {

using namespace viennacore;

// Simple geometric model that implements a
template <typename NumericType, int D, typename DistType>
class GeometricDistributionModel : public GeometricModel<NumericType, D> {
  static_assert(
      std::is_base_of_v<viennals::GeometricAdvectDistribution<hrleCoordType, D>,
                        DistType>);

  using GeomDistPtr = SmartPointer<DistType>;
  using LSPtr = SmartPointer<viennals::Domain<NumericType, D>>;

  using GeometricModel<NumericType, D>::domain;

  GeomDistPtr dist = nullptr;
  LSPtr mask = nullptr;

public:
  GeometricDistributionModel(GeomDistPtr passedDist) : dist(passedDist) {}

  GeometricDistributionModel(GeomDistPtr passedDist, LSPtr passedMask)
      : dist(passedDist), mask(passedMask) {}

  void apply() {
    if (dist) {
      if (mask) {
        viennals::GeometricAdvect<NumericType, D>(
            domain->getLevelSets()->back(), dist, mask)
            .apply();
      } else {
        viennals::GeometricAdvect<NumericType, D>(
            domain->getLevelSets()->back(), dist)
            .apply();
      }
    }
  }
};

template <typename NumericType, int D>
class SphereDistribution : public ProcessModel<NumericType, D> {
  using LSPtr = SmartPointer<viennals::Domain<NumericType, D>>;

public:
  SphereDistribution(const NumericType radius, const NumericType gridDelta,
                     LSPtr mask = nullptr) {
    auto dist =
        SmartPointer<viennals::SphereDistribution<hrleCoordType, D>>::New(
            radius, gridDelta);

    auto geomModel = SmartPointer<GeometricDistributionModel<
        NumericType, D,
        viennals::SphereDistribution<hrleCoordType, D>>>::New(dist, mask);

    this->setGeometricModel(geomModel);
    this->setProcessName("SphereDistribution");
  }
};

template <typename NumericType, int D>
class BoxDistribution : public ProcessModel<NumericType, D> {
  using LSPtr = SmartPointer<viennals::Domain<NumericType, D>>;

public:
  BoxDistribution(const std::array<hrleCoordType, 3> &halfAxes,
                  const NumericType gridDelta, LSPtr mask = nullptr) {
    auto dist = SmartPointer<viennals::BoxDistribution<hrleCoordType, D>>::New(
        halfAxes, gridDelta);

    auto geomModel = SmartPointer<GeometricDistributionModel<
        NumericType, D,
        viennals::BoxDistribution<hrleCoordType, D>>>::New(dist, mask);

    this->setGeometricModel(geomModel);
    this->setProcessName("BoxDistribution");
  }
};

} // namespace viennaps
