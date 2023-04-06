#pragma once

#include <lsBooleanOperation.hpp>
#include <lsDomain.hpp>
#include <lsMakeGeometry.hpp>
#include <psDomain.hpp>

/**
 * Creates a fin gemeotry in in z(3D)/y(2D) direction.
 */
template <class NumericType, int D> class psMakeFin {
  using LSPtrType = psSmartPointer<lsDomain<NumericType, D>>;
  using PSPtrType = psSmartPointer<psDomain<NumericType, D>>;

public:
  PSPtrType domain = nullptr;

  NumericType gridDelta = .25;
  NumericType xExtent = 10;
  NumericType yExtent = 7;
  NumericType baseHeight = 0.;

  NumericType finWidth = 7;
  NumericType finHeight = 17.5;
  bool periodicBoundary = false;
  bool makeMask = true;

  psMakeFin(PSPtrType passedDomain) : domain(passedDomain) {}

  psMakeFin(PSPtrType passedDomain, const NumericType passedGridDelta,
            const NumericType passedXExtent, const NumericType passedYExtent,
            const NumericType passedFinWidth, const NumericType passedFinHeight,
            const NumericType passedBaseHeight = 0.,
            const bool passedPeriodic = false, const bool passedMakeMask = true)
      : domain(passedDomain), gridDelta(passedGridDelta),
        xExtent(passedXExtent), yExtent(passedYExtent),
        finWidth(passedFinWidth), finHeight(passedFinHeight),
        baseHeight(passedBaseHeight), periodicBoundary(passedPeriodic),
        makeMask(passedMakeMask) {}

  void apply() {
    domain->clear();
    if constexpr (D == 3) {

      double bounds[2 * D] = {-xExtent / 2, xExtent / 2, -yExtent / 2,
                              yExtent / 2,  -gridDelta,  finHeight + gridDelta};

      typename lsDomain<NumericType, D>::BoundaryType boundaryCons[3] = {
          lsDomain<NumericType, D>::BoundaryType::REFLECTIVE_BOUNDARY,
          lsDomain<NumericType, D>::BoundaryType::REFLECTIVE_BOUNDARY,
          lsDomain<NumericType, D>::BoundaryType::INFINITE_BOUNDARY};

      auto substrate = LSPtrType::New(bounds, boundaryCons, gridDelta);
      NumericType normal[D] = {0., 0., 1.};
      NumericType origin[D] = {0., 0., 0.};
      lsMakeGeometry<NumericType, D>(
          substrate,
          lsSmartPointer<lsPlane<NumericType, D>>::New(origin, normal))
          .apply();

      auto mask = LSPtrType::New(bounds, boundaryCons, gridDelta);

      NumericType minPoint[D] = {-finWidth / 2, -yExtent / 2, -gridDelta};
      NumericType maxPoint[D] = {finWidth / 2, yExtent / 2, finHeight};

      lsMakeGeometry<NumericType, D>(
          mask, lsSmartPointer<lsBox<NumericType, D>>::New(minPoint, maxPoint))
          .apply();

      lsBooleanOperation<NumericType, D>(substrate, mask,
                                         lsBooleanOperationEnum::UNION)
          .apply();
      if (makeMask)
        domain->insertNextLevelSet(mask);
      domain->insertNextLevelSet(substrate, false);
    } else if constexpr (D == 2) {

      double bounds[2 * D] = {-xExtent / 2, xExtent / 2, -gridDelta,
                              finHeight + gridDelta};

      typename lsDomain<NumericType, D>::BoundaryType boundaryCons[2] = {
          lsDomain<NumericType, D>::BoundaryType::REFLECTIVE_BOUNDARY,
          lsDomain<NumericType, D>::BoundaryType::INFINITE_BOUNDARY};

      auto substrate = LSPtrType::New(bounds, boundaryCons, gridDelta);
      NumericType normal[D] = {0., 1.};
      NumericType origin[D] = {0., 0.};
      lsMakeGeometry<NumericType, D>(
          substrate,
          lsSmartPointer<lsPlane<NumericType, D>>::New(origin, normal))
          .apply();

      auto mask = LSPtrType::New(bounds, boundaryCons, gridDelta);
      NumericType minPoint[D] = {-finWidth / 2, -gridDelta};
      NumericType maxPoint[D] = {finWidth / 2, finHeight};
      lsMakeGeometry<NumericType, D>(
          mask, lsSmartPointer<lsBox<NumericType, D>>::New(minPoint, maxPoint))
          .apply();

      lsBooleanOperation<NumericType, D>(substrate, mask,
                                         lsBooleanOperationEnum::UNION)
          .apply();
      if (makeMask)
        domain->insertNextLevelSet(mask);
      domain->insertNextLevelSet(substrate, false);
    }
  }
};