#pragma once

#include "psDomain.hpp"

#include <lsBooleanOperation.hpp>
#include <lsMakeGeometry.hpp>

namespace viennaps {

using namespace viennacore;

template <class NumericType, int D> class Planarize {
  SmartPointer<Domain<NumericType, D>> pDomain_;
  NumericType cutoffPosition_ = 0.;

public:
  Planarize() = default;
  Planarize(SmartPointer<Domain<NumericType, D>> domain,
            const NumericType passedCutoff)
      : pDomain_(domain), cutoffPosition_(passedCutoff) {}

  void setDomain(SmartPointer<Domain<NumericType, D>> domain) {
    pDomain_ = domain;
  }

  void setCutoffPosition(const NumericType passedCutoff) {
    cutoffPosition_ = passedCutoff;
  }

  void apply() {
    NumericType origin[D] = {0.};
    origin[D - 1] = cutoffPosition_;
    NumericType normal[D] = {0.};
    normal[D - 1] = -1.;
    auto plane = viennals::Domain<NumericType, D>::New(pDomain_->getGrid());
    viennals::MakeGeometry<NumericType, D>(
        plane, viennals::Plane<NumericType, D>::New(origin, normal))
        .apply();
    pDomain_->applyBooleanOperation(
        plane, viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT);
  }
};

} // namespace viennaps
