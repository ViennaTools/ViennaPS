#pragma once

#include <lsBooleanOperation.hpp>
#include <lsDomain.hpp>
#include <lsMakeGeometry.hpp>

#include <psDomain.hpp>

/**
  Creates a trench geometry in z(3D)/y(2D) direction.
*/
template <class NumericType, int D> class psMakeTrench {
  using LSPtrType = psSmartPointer<lsDomain<NumericType, D>>;
  using PSPtrType = psSmartPointer<psDomain<NumericType, D>>;

public:
  PSPtrType domain = nullptr;

  NumericType gridDelta = .25;
  NumericType xExtent = 20;
  NumericType yExtent = 14;

  NumericType trenchWidth = 7;
  NumericType trenchDepth = 17.5;
  bool makeMask = true;

  psMakeTrench(PSPtrType passedDomain) : domain(passedDomain) {}

  psMakeTrench(PSPtrType passedDomain, const NumericType passedGridDelta,
               const NumericType passedXExtent, const NumericType passedYExtent,
               const NumericType passedTrenchWidth,
               const NumericType passedTrenchHeight,
               const bool passedMakeMask = true)
      : domain(passedDomain), gridDelta(passedGridDelta),
        xExtent(passedXExtent), yExtent(passedYExtent),
        trenchWidth(passedTrenchWidth), trenchDepth(passedTrenchHeight),
        makeMask(passedMakeMask) {}

  void apply() {
    domain->clear();
    double bounds[2 * D];
    bounds[0] = -xExtent / 2.;
    bounds[1] = xExtent / 2.;

    if constexpr (D == 3) {
      bounds[2] = -yExtent / 2.;
      bounds[3] = yExtent / 2.;
      bounds[4] = -gridDelta;
      bounds[5] = trenchDepth + gridDelta;
    } else {
      bounds[2] = -gridDelta;
      bounds[3] = trenchDepth + gridDelta;
    }

    typename lsDomain<NumericType, D>::BoundaryType boundaryCons[D];

    for (int i = 0; i < D - 1; i++)
      boundaryCons[i] =
          lsDomain<NumericType, D>::BoundaryType::REFLECTIVE_BOUNDARY;
    boundaryCons[D - 1] =
        lsDomain<NumericType, D>::BoundaryType::INFINITE_BOUNDARY;

    auto substrate = LSPtrType::New(bounds, boundaryCons, gridDelta);
    NumericType normal[D] = {0.};
    NumericType origin[D] = {0.};
    normal[D - 1] = 1.;
    lsMakeGeometry<NumericType, D>(
        substrate, lsSmartPointer<lsPlane<NumericType, D>>::New(origin, normal))
        .apply();

    auto mask = LSPtrType::New(bounds, boundaryCons, gridDelta);
    origin[D - 1] = trenchDepth;
    lsMakeGeometry<NumericType, D>(
        mask, lsSmartPointer<lsPlane<NumericType, D>>::New(origin, normal))
        .apply();

    auto maskAdd = LSPtrType::New(bounds, boundaryCons, gridDelta);
    origin[D - 1] = 0.;
    normal[D - 1] = -1.;
    lsMakeGeometry<NumericType, D>(
        maskAdd, lsSmartPointer<lsPlane<NumericType, D>>::New(origin, normal))
        .apply();

    lsBooleanOperation<NumericType, D>(mask, maskAdd,
                                       lsBooleanOperationEnum::INTERSECT)
        .apply();

    NumericType minPoint[D];
    NumericType maxPoint[D];

    minPoint[0] = -trenchWidth / 2;
    maxPoint[0] = trenchWidth / 2;

    if constexpr (D == 3) {
      minPoint[1] = -yExtent / 2.;
      maxPoint[1] = yExtent / 2.;
      minPoint[2] = -gridDelta;
      maxPoint[2] = trenchDepth;
    } else {
      minPoint[1] = -gridDelta;
      maxPoint[1] = trenchDepth;
    }

    lsMakeGeometry<NumericType, D>(
        maskAdd, lsSmartPointer<lsBox<NumericType, D>>::New(minPoint, maxPoint))
        .apply();

    lsBooleanOperation<NumericType, D>(
        mask, maskAdd, lsBooleanOperationEnum::RELATIVE_COMPLEMENT)
        .apply();

    lsBooleanOperation<NumericType, D>(substrate, mask,
                                       lsBooleanOperationEnum::UNION)
        .apply();

    if (makeMask)
      domain->insertNextLevelSet(mask);
    domain->insertNextLevelSet(substrate, false);
  }
};