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
  static_assert(std::is_base_of_v<
                lsGeometricAdvectDistribution<hrleCoordType, D>, DistType>);

  using GeomDistPtr = lsSmartPointer<DistType>;
  using LSPtr = lsSmartPointer<lsDomain<NumericType, D>>;

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
        lsGeometricAdvect<NumericType, D>(domain->getLevelSets()->back(), dist,
                                          mask)
            .apply();
      } else {
        lsGeometricAdvect<NumericType, D>(domain->getLevelSets()->back(), dist)
            .apply();
      }
    }
  }
};

template <typename NumericType, int D>
class SphereDistribution : public ProcessModel<NumericType, D> {
  using LSPtr = lsSmartPointer<lsDomain<NumericType, D>>;

public:
  SphereDistribution(const NumericType radius, const NumericType gridDelta,
                     LSPtr mask = nullptr) {
    auto dist = lsSmartPointer<lsSphereDistribution<hrleCoordType, D>>::New(
        radius, gridDelta);

    auto geomModel = lsSmartPointer<GeometricDistributionModel<
        NumericType, D, lsSphereDistribution<hrleCoordType, D>>>::New(dist,
                                                                      mask);

    this->setGeometricModel(geomModel);
    this->setProcessName("SphereDistribution");
  }
};

template <typename NumericType, int D>
class BoxDistribution : public ProcessModel<NumericType, D> {
  using LSPtr = lsSmartPointer<lsDomain<NumericType, D>>;

public:
  BoxDistribution(const std::array<hrleCoordType, 3> &halfAxes,
                  const NumericType gridDelta, LSPtr mask = nullptr) {
    auto dist = lsSmartPointer<lsBoxDistribution<hrleCoordType, D>>::New(
        halfAxes, gridDelta);

    auto geomModel = lsSmartPointer<GeometricDistributionModel<
        NumericType, D, lsBoxDistribution<hrleCoordType, D>>>::New(dist, mask);

    this->setGeometricModel(geomModel);
    this->setProcessName("BoxDistribution");
  }
};

} // namespace viennaps
