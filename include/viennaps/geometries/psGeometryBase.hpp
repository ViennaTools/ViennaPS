#pragma once

#include "../psDomain.hpp"

#include <lsBooleanOperation.hpp>
#include <lsMakeGeometry.hpp>

namespace viennaps {

using namespace viennacore;

template <class NumericType, int D> class GeometryBase {
protected:
  using lsDomainType = SmartPointer<viennals::Domain<NumericType, D>>;
  using psDomainType = SmartPointer<Domain<NumericType, D>>;

  psDomainType domain_ = nullptr;

public:
  GeometryBase(psDomainType domain) : domain_(domain) {}

  lsDomainType makeSubstrate(NumericType base) {
    auto gridDelta = domain_->getSetup().gridDelta_;
    auto bounds = domain_->getSetup().bounds_;
    auto boundaryCons = domain_->getSetup().boundaryCons_;
    assert(gridDelta > 0.);

    auto substrate = lsDomainType::New(bounds, boundaryCons, gridDelta);

    NumericType normal[D] = {0.};
    NumericType origin[D] = {0.};

    normal[D - 1] = 1.;
    origin[D - 1] = base;
    viennals::MakeGeometry<NumericType, D>(
        substrate,
        SmartPointer<viennals::Plane<NumericType, D>>::New(origin, normal))
        .apply();

    return substrate;
  }

  lsDomainType makeMask(NumericType base, NumericType height) {
    auto gridDelta = domain_->getSetup().gridDelta_;
    auto bounds = domain_->getSetup().bounds_;
    auto boundaryCons = domain_->getSetup().boundaryCons_;
    assert(gridDelta > 0.);

    auto mask = lsDomainType::New(bounds, boundaryCons, gridDelta);

    NumericType normal[D] = {0.};
    NumericType origin[D] = {0.};

    normal[D - 1] = 1.;
    origin[D - 1] = base + height;
    viennals::MakeGeometry<NumericType, D>(
        mask,
        SmartPointer<viennals::Plane<NumericType, D>>::New(origin, normal))
        .apply();

    auto maskAdd = lsDomainType::New(bounds, boundaryCons, gridDelta);
    origin[D - 1] = base - gridDelta / 2.;
    normal[D - 1] = -1.;
    viennals::MakeGeometry<NumericType, D>(
        maskAdd,
        SmartPointer<viennals::Plane<NumericType, D>>::New(origin, normal))
        .apply();

    viennals::BooleanOperation<NumericType, D>(
        mask, maskAdd, viennals::BooleanOperationEnum::INTERSECT)
        .apply();

    return mask;
  }
};

} // namespace viennaps
