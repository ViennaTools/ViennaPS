#pragma once

#include "psDomain.hpp"

#include <lsBooleanOperation.hpp>
#include <lsMakeGeometry.hpp>

template <class NumericType, int D> class psPlanarize {
  psSmartPointer<psDomain<NumericType, D>> pDomain_;
  NumericType cutoffPosition_ = 0.;

public:
  psPlanarize() {}
  psPlanarize(psSmartPointer<psDomain<NumericType, D>> domain,
              const NumericType passedCutoff)
      : pDomain_(domain), cutoffPosition_(passedCutoff) {}

  void setDomain(psSmartPointer<psDomain<NumericType, D>> domain) {
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
    auto plane =
        lsSmartPointer<lsDomain<NumericType, D>>::New(pDomain_->getGrid());
    lsMakeGeometry<NumericType, D>(
        plane, lsSmartPointer<lsPlane<NumericType, D>>::New(origin, normal))
        .apply();
    pDomain_->applyBooleanOperation(
        ls, lsBooleanOperationEnum::RELATIVE_COMPLEMENT);
  }
};
