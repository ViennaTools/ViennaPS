#ifndef GEOMETRIC_DISTRIBUTION_MODEL_HPP
#define GEOMETRIC_DISTRIBUTION_MODEL_HPP

#include <type_traits>

#include <lsGeometricAdvect.hpp>
#include <lsGeometricAdvectDistributions.hpp>

#include <psGeometricModel.hpp>
#include <psProcessModel.hpp>
#include <psSmartPointer.hpp>

// Simple geometric model that implements a
template <typename NumericType, int D, typename DistType>
class GeometricDistributionModel : public psGeometricModel<NumericType, D> {
  static_assert(std::is_base_of_v<
                lsGeometricAdvectDistribution<hrleCoordType, D>, DistType>);

  using GeomDistPtr = psSmartPointer<DistType>;
  using LSPtr = psSmartPointer<lsDomain<NumericType, D>>;

  using psGeometricModel<NumericType, D>::domain;

  GeomDistPtr dist = nullptr;
  LSPtr mask = nullptr;

public:
  GeometricDistributionModel(GeomDistPtr passedDist) : dist(passedDist) {}

  GeometricDistributionModel(GeomDistPtr passedDist, LSPtr passedMask)
      : dist(passedDist), mask(passedMask) {}

  void apply() {
    if (dist)
      if (mask)
        lsGeometricAdvect<NumericType, D>(domain->getLevelSets()->back(), dist,
                                          mask)
            .apply();
      else
        lsGeometricAdvect<NumericType, D>(domain->getLevelSets()->back(), dist)
            .apply();
  }
};

template <typename NumericType, int D> class SphereDistribution {
  using LSPtr = psSmartPointer<lsDomain<NumericType, D>>;

  psSmartPointer<psProcessModel<NumericType, D>> processModel = nullptr;

public:
  SphereDistribution(const NumericType radius, const NumericType gridDelta,
                     LSPtr mask = nullptr) {
    processModel = psSmartPointer<psProcessModel<NumericType, D>>::New();

    auto dist = psSmartPointer<lsSphereDistribution<hrleCoordType, D>>::New(
        radius, gridDelta);

    auto geomModel = psSmartPointer<GeometricDistributionModel<
        NumericType, D, lsSphereDistribution<hrleCoordType, D>>>::New(dist,
                                                                      mask);

    processModel->setGeometricModel(geomModel);
    processModel->setProcessName("SphereDistribution");
  }

  psSmartPointer<psProcessModel<NumericType, D>> getProcessModel() {
    return processModel;
  }
};

template <typename NumericType, int D> class BoxDistribution {
  using LSPtr = psSmartPointer<lsDomain<NumericType, D>>;

  psSmartPointer<psProcessModel<NumericType, D>> processModel = nullptr;

public:
  BoxDistribution(const std::array<hrleCoordType, 3> &halfAxes,
                  const NumericType gridDelta, LSPtr mask = nullptr) {
    processModel = psSmartPointer<psProcessModel<NumericType, D>>::New();

    auto dist = psSmartPointer<lsBoxDistribution<hrleCoordType, D>>::New(
        halfAxes, gridDelta);

    auto geomModel = psSmartPointer<GeometricDistributionModel<
        NumericType, D, lsBoxDistribution<hrleCoordType, D>>>::New(dist, mask);

    processModel->setGeometricModel(geomModel);
    processModel->setProcessName("BoxDistribution");
  }

  psSmartPointer<psProcessModel<NumericType, D>> getProcessModel() {
    return processModel;
  }
};
#endif