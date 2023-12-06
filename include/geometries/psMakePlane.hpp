#pragma once

#include <lsDomain.hpp>
#include <lsMakeGeometry.hpp>

#include <psDomain.hpp>
#include <psLogger.hpp>
#include <psMaterials.hpp>

/// This class provides a simple way to create a plane in a level set. It can be
/// used to create a substrate of any material. The plane can be added to an
/// already existing geometry or a new geometry can be created. The plane is
/// created with normal direction in the positive z direction in 3D and positive
/// y direction in 2D. The plane is centered around the origin with the total
/// specified extent and height. The plane can have a periodic boundary in the x
/// and y (only 3D) direction.
template <class NumericType, int D> class psMakePlane {
  using LSPtrType = psSmartPointer<lsDomain<NumericType, D>>;
  using psDomainType = psSmartPointer<psDomain<NumericType, D>>;

  psDomainType domain = nullptr;

  const NumericType gridDelta = 0.;
  const NumericType xExtent = 0.;
  const NumericType yExtent = 0.;
  const NumericType height = 0.;

  const bool periodicBoundary = false;
  const bool add = false;

  const psMaterial material = psMaterial::None;

public:
  psMakePlane(psDomainType passedDomain, NumericType passedHeight = 0.,
              const psMaterial passedMaterial = psMaterial::None)
      : domain(passedDomain), height(passedHeight), add(true),
        material(passedMaterial) {}

  psMakePlane(psDomainType passedDomain, const NumericType passedGridDelta,
              const NumericType passedXExtent, const NumericType passedYExtent,
              const NumericType passedHeight, const bool passedPeriodic = false,
              const psMaterial passedMaterial = psMaterial::None)
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
      if (material == psMaterial::None) {
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
      if (material == psMaterial::None) {
        domain->insertNextLevelSet(substrate);
      } else {
        domain->insertNextLevelSetAsMaterial(substrate, material);
      }
    }
  }
};