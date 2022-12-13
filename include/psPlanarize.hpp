#pragma once

#include <psDomain.hpp>

#include <lsBooleanOperation.hpp>
#include <lsMakeGeometry.hpp>

template <class NumericType, int D> class psPlanarize {
  psSmartPointer<psDomain<NumericType, D>> domain;
  NumericType cutoffPosition = 0.;

public:
  psPlanarize() {}
  psPlanarize(psSmartPointer<psDomain<NumericType, D>> passedDomain,
              const NumericType passedCutoff)
      : domain(passedDomain), cutoffPosition(passedCutoff) {}

  void apply() {
    NumericType origin[D] = {0.};
    origin[D - 1] = cutoffPosition;
    NumericType normal[D] = {0.};
    normal[D - 1] = -1.;
    auto plane = lsSmartPointer<lsDomain<NumericType, D>>::New(
        domain->getLevelSets()->back()->getGrid());
    lsMakeGeometry<NumericType, D>(
        plane, lsSmartPointer<lsPlane<NumericType, D>>::New(origin, normal))
        .apply();
    for (auto &ls : *domain->getLevelSets()) {
      lsBooleanOperation<NumericType, D>(
          ls, plane, lsBooleanOperationEnum::RELATIVE_COMPLEMENT)
          .apply();
    }
  }
};