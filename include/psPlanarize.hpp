#pragma once

#include <psDomain.hpp>

#include <lsBooleanOperation.hpp>
#include <lsMakeGeometry.hpp>

/// Planarizes the domain at the specified cutoff position. The planarization
/// process involves subtracting a plane from all materials within the domain
/// using a boolean operation.
/// Example usage:
/// \code{.cpp}
///   psPlanarize<double, 3>(domain, 0.).apply();
/// \endcode
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
    domain->applyBooleanOperation(plane,
                                  lsBooleanOperationEnum::RELATIVE_COMPLEMENT);
  }
};
