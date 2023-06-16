#pragma once

#include <lsDomain.hpp>
#include <lsMakeGeometry.hpp>

#include <psDomain.hpp>
#include <psLogger.hpp>
#include <psMaterials.hpp>

/**
 * Creates a plane in z(3D)/y(2D) direction.
 */
template <class NumericType, int D> class psMakePlane {
  using LSPtrType = psSmartPointer<lsDomain<NumericType, D>>;
  using PSPtrType = psSmartPointer<psDomain<NumericType, D>>;

  PSPtrType domain = nullptr;

  const NumericType gridDelta = 0.;
  const NumericType xExtent = 0.;
  const NumericType yExtent = 0.;
  const NumericType height = 0.;

  const bool periodicBoundary = false;
  const bool add = false;

  const psMaterial material = psMaterial::Undefined;

public:
  psMakePlane(PSPtrType passedDomain, NumericType passedHeight = 0.,
              const psMaterial passedMaterial = psMaterial::Undefined)
      : domain(passedDomain), height(passedHeight), add(true),
        material(passedMaterial) {}

  psMakePlane(PSPtrType passedDomain, const NumericType passedGridDelta,
              const NumericType passedXExtent, const NumericType passedYExtent,
              const NumericType passedHeight, const bool passedPeriodic = false,
              const psMaterial passedMaterial = psMaterial::Undefined)
      : domain(passedDomain), gridDelta(passedGridDelta),
        xExtent(passedXExtent), yExtent(passedYExtent), height(passedHeight),
        periodicBoundary(passedPeriodic), material(passedMaterial) {}

  void apply() {
    if (add) {
      if (!domain->getLevelSets()->back()) {
        psLogger::getInstance()
            .addWarning("psMakePlane: Plane can only be added to already "
                        "existing geometry.")
            .print();
        return;
      }
    } else {
      domain->clear();
    }

    double bounds[2 * D];
    bounds[0] = -xExtent / 2.;
    bounds[1] = xExtent / 2.;

    if constexpr (D == 3) {
      bounds[2] = -yExtent / 2.;
      bounds[3] = yExtent / 2.;
      bounds[4] = -gridDelta;
      bounds[5] = gridDelta;
    } else {
      bounds[2] = -gridDelta;
      bounds[3] = gridDelta;
    }

    typename lsDomain<NumericType, D>::BoundaryType boundaryCons[D];

    for (int i = 0; i < D - 1; i++) {
      if (periodicBoundary) {
        boundaryCons[i] =
            lsDomain<NumericType, D>::BoundaryType::PERIODIC_BOUNDARY;
      } else {
        boundaryCons[i] =
            lsDomain<NumericType, D>::BoundaryType::REFLECTIVE_BOUNDARY;
      }
    }
    boundaryCons[D - 1] =
        lsDomain<NumericType, D>::BoundaryType::INFINITE_BOUNDARY;

    NumericType normal[D] = {0.};
    NumericType origin[D] = {0.};
    normal[D - 1] = 1.;
    origin[D - 1] = height;

    if (add) {
      auto substrate =
          LSPtrType::New(domain->getLevelSets()->back()->getGrid());
      lsMakeGeometry<NumericType, D>(
          substrate,
          lsSmartPointer<lsPlane<NumericType, D>>::New(origin, normal))
          .apply();
      if (material == psMaterial::Undefined) {
        domain->insertNextLevelSet(substrate);
      } else {
        domain->insertNextLevelSetAsMaterial(substrate, material);
      }
    } else {
      auto substrate = LSPtrType::New(bounds, boundaryCons, gridDelta);
      lsMakeGeometry<NumericType, D>(
          substrate,
          lsSmartPointer<lsPlane<NumericType, D>>::New(origin, normal))
          .apply();
      if (material == psMaterial::Undefined) {
        domain->insertNextLevelSet(substrate);
      } else {
        domain->insertNextLevelSetAsMaterial(substrate, material);
      }
    }
  }
};