#pragma once

#include <type_traits>

#include <lsGeometricAdvect.hpp>
#include <lsGeometricAdvectDistributions.hpp>

#include "../psGeometricModel.hpp"
#include "../psProcessModel.hpp"

namespace viennaps {

using namespace viennacore;

template <typename NumericType, int D, typename DistType>
class GeometricDistributionModel : public GeometricModel<NumericType, D> {
  static_assert(std::is_base_of_v<
                viennals::GeometricAdvectDistribution<viennahrle::CoordType, D>,
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

  void apply() override {
    if (dist) {
      if (static_cast<int>(domain->getMetaDataLevel()) > 1) {
        domain->clearMetaData();
        domain->addMetaData(this->processData);
      }
      if (mask) {
        viennals::GeometricAdvect<NumericType, D>(domain->getLevelSets().back(),
                                                  dist, mask)
            .apply();
      } else {
        viennals::GeometricAdvect<NumericType, D>(domain->getLevelSets().back(),
                                                  dist)
            .apply();
      }
    }
  }
};

template <typename NumericType, int D>
class SphereDistribution : public ProcessModel<NumericType, D> {
  using LSPtr = SmartPointer<viennals::Domain<NumericType, D>>;

public:
  SphereDistribution(NumericType radius, NumericType gridDelta,
                     LSPtr mask = nullptr) {
    auto dist = SmartPointer<
        viennals::SphereDistribution<viennahrle::CoordType, D>>::New(radius,
                                                                     gridDelta);

    auto geomModel = SmartPointer<GeometricDistributionModel<
        NumericType, D,
        viennals::SphereDistribution<viennahrle::CoordType, D>>>::New(dist,
                                                                      mask);

    this->setGeometricModel(geomModel);
    this->setProcessName("SphereDistribution");
    this->processMetaData["Radius"] = std::vector<NumericType>{radius};
  }
};

template <typename NumericType, int D>
class BoxDistribution : public ProcessModel<NumericType, D> {
  using LSPtr = SmartPointer<viennals::Domain<NumericType, D>>;

public:
  BoxDistribution(const std::array<viennahrle::CoordType, 3> &halfAxes,
                  NumericType gridDelta, LSPtr mask = nullptr) {
    auto dist =
        SmartPointer<viennals::BoxDistribution<viennahrle::CoordType, D>>::New(
            halfAxes, gridDelta);

    auto geomModel = SmartPointer<GeometricDistributionModel<
        NumericType, D,
        viennals::BoxDistribution<viennahrle::CoordType, D>>>::New(dist, mask);

    this->setGeometricModel(geomModel);
    this->setProcessName("BoxDistribution");
    this->processMetaData["HalfAxes"] =
        std::vector<NumericType>{static_cast<NumericType>(halfAxes[0]),
                                 static_cast<NumericType>(halfAxes[1]),
                                 static_cast<NumericType>(halfAxes[2])};
  }
};

PS_PRECOMPILE_PRECISION_DIMENSION(SphereDistribution)
PS_PRECOMPILE_PRECISION_DIMENSION(BoxDistribution)

} // namespace viennaps
