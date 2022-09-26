#pragma once

#include <lsBooleanOperation.hpp>
#include <lsDomain.hpp>
#include <lsMakeGeometry.hpp>
#include <lsToSurfaceMesh.hpp>
#include <lsVTKWriter.hpp>
#include <lsWriteVisualizationMesh.hpp>
#include <psDomain.hpp>
#include <string>

template <class NumericType, int D> class psMakeTrench {
  using LSPtrType = psSmartPointer<lsDomain<NumericType, D>>;
  using PSPtrType = psSmartPointer<psDomain<NumericType, D>>;

public:
  PSPtrType domain = nullptr;

  NumericType gridDelta = .25;
  NumericType xExtent = 10;
  NumericType yExtent = 7;

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
    if constexpr (D == 3) {
      double bounds[2 * D] = {-xExtent, xExtent,    -yExtent,
                              yExtent,  -gridDelta, trenchDepth + gridDelta};

      typename lsDomain<NumericType, D>::BoundaryType boundaryCons[3] = {
          lsDomain<NumericType, D>::BoundaryType::PERIODIC_BOUNDARY,
          lsDomain<NumericType, D>::BoundaryType::PERIODIC_BOUNDARY,
          lsDomain<NumericType, D>::BoundaryType::INFINITE_BOUNDARY};

      auto substrate = LSPtrType::New(bounds, boundaryCons, gridDelta);
      NumericType normal[D] = {0., 0., 1.};
      NumericType origin[D] = {0., 0., 0.};
      lsMakeGeometry<NumericType, D>(
          substrate,
          lsSmartPointer<lsPlane<NumericType, D>>::New(origin, normal))
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

      NumericType minPoint[D] = {-trenchWidth / 2, -yExtent, -gridDelta};
      NumericType maxPoint[D] = {trenchWidth / 2, yExtent, trenchDepth};
      lsMakeGeometry<NumericType, D>(
          maskAdd,
          lsSmartPointer<lsBox<NumericType, D>>::New(minPoint, maxPoint))
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
    } else if constexpr (D == 2) {
      double bounds[2 * D] = {-xExtent, xExtent, -gridDelta,
                              trenchDepth + gridDelta};

      typename lsDomain<NumericType, D>::BoundaryType boundaryCons[2] = {
          lsDomain<NumericType, D>::BoundaryType::PERIODIC_BOUNDARY,
          lsDomain<NumericType, D>::BoundaryType::INFINITE_BOUNDARY};

      auto substrate = LSPtrType::New(bounds, boundaryCons, gridDelta);
      NumericType normal[D] = {0., 1.};
      NumericType origin[D] = {0., 0.};
      lsMakeGeometry<NumericType, D>(
          substrate,
          lsSmartPointer<lsPlane<NumericType, D>>::New(origin, normal))
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

      NumericType minPoint[D] = {-trenchWidth / 2, -gridDelta};
      NumericType maxPoint[D] = {trenchWidth / 2, trenchDepth};
      lsMakeGeometry<NumericType, D>(
          maskAdd,
          lsSmartPointer<lsBox<NumericType, D>>::New(minPoint, maxPoint))
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
  }
};