#pragma once

#include <lsBooleanOperation.hpp>
#include <lsDomain.hpp>
#include <lsMakeGeometry.hpp>
#include <psDomain.hpp>

/// Generates a new fin geometry extending in the z (3D) or y (2D) direction,
/// centered at the origin with specified dimensions in the x and y directions.
/// The fin may incorporate periodic boundaries in the x and y directions
/// (limited to 3D). Users can define the width and height of the fin, and it
/// can function as a mask, with the specified material exclusively applied to
/// the bottom of the fin, while the upper portion adopts the mask material.
template <class NumericType, int D> class psMakeFin {
  using LSPtrType = psSmartPointer<lsDomain<NumericType, D>>;
  using psDomainType = psSmartPointer<psDomain<NumericType, D>>;

  psDomainType domain = nullptr;

  const NumericType gridDelta;
  const NumericType xExtent;
  const NumericType yExtent;
  const NumericType baseHeight = 0.;
  const NumericType taperAngle = 0; // tapering angle in degrees

  const NumericType finWidth;
  const NumericType finHeight;
  const bool periodicBoundary = false;
  const bool makeMask = false;
  const psMaterial material = psMaterial::None;

public:
  psMakeFin(psDomainType passedDomain, const NumericType passedGridDelta,
            const NumericType passedXExtent, const NumericType passedYExtent,
            const NumericType passedFinWidth, const NumericType passedFinHeight,
            const NumericType passedTaperAngle = 0.,
            const NumericType passedBaseHeight = 0.,
            const bool passedPeriodic = false,
            const bool passedMakeMask = false,
            const psMaterial passedMaterial = psMaterial::None)
      : domain(passedDomain), gridDelta(passedGridDelta),
        xExtent(passedXExtent), yExtent(passedYExtent),
        finWidth(passedFinWidth), finHeight(passedFinHeight),
        taperAngle(passedTaperAngle), baseHeight(passedBaseHeight),
        periodicBoundary(passedPeriodic), makeMask(passedMakeMask),
        material(passedMaterial) {}

  void apply() {
    domain->clear();
    if constexpr (D == 3) {

      double bounds[2 * D] = {
          -xExtent / 2,           xExtent / 2,
          -yExtent / 2,           yExtent / 2,
          baseHeight - gridDelta, baseHeight + finHeight + gridDelta};

      typename lsDomain<NumericType, D>::BoundaryType boundaryCons[D] = {
          lsDomain<NumericType, D>::BoundaryType::REFLECTIVE_BOUNDARY,
          lsDomain<NumericType, D>::BoundaryType::REFLECTIVE_BOUNDARY,
          lsDomain<NumericType, D>::BoundaryType::INFINITE_BOUNDARY};
      if (periodicBoundary) {
        boundaryCons[0] =
            lsDomain<NumericType, D>::BoundaryType::PERIODIC_BOUNDARY;
        boundaryCons[1] =
            lsDomain<NumericType, D>::BoundaryType::PERIODIC_BOUNDARY;
      }

      auto substrate = LSPtrType::New(bounds, boundaryCons, gridDelta);
      NumericType normal[D] = {0., 0., 1.};
      NumericType origin[D] = {0., 0., baseHeight};
      lsMakeGeometry<NumericType, D>(
          substrate,
          lsSmartPointer<lsPlane<NumericType, D>>::New(origin, normal))
          .apply();

      auto mask = LSPtrType::New(bounds, boundaryCons, gridDelta);

      if (taperAngle == 0.) {
        NumericType minPoint[D] = {-finWidth / 2, -yExtent / 2,
                                   baseHeight - gridDelta};
        NumericType maxPoint[D] = {finWidth / 2, yExtent / 2,
                                   baseHeight + finHeight};

        lsMakeGeometry<NumericType, D>(
            mask,
            lsSmartPointer<lsBox<NumericType, D>>::New(minPoint, maxPoint))
            .apply();

        lsBooleanOperation<NumericType, D>(substrate, mask,
                                           lsBooleanOperationEnum::UNION)
            .apply();
      } else {
        psLogger::getInstance()
            .addWarning(
                "psMakeFin: Tapered fins are not yet implemented in 3D!")
            .print();
      }

      if (material == psMaterial::None) {
        if (makeMask)
          domain->insertNextLevelSet(mask);
        domain->insertNextLevelSet(substrate, false);
      } else {
        if (makeMask)
          domain->insertNextLevelSetAsMaterial(mask, psMaterial::Mask);
        domain->insertNextLevelSetAsMaterial(substrate, material, false);
      }
    } else if constexpr (D == 2) {

      double bounds[2 * D] = {-xExtent / 2, xExtent / 2, baseHeight - gridDelta,
                              baseHeight + finHeight + gridDelta};

      typename lsDomain<NumericType, D>::BoundaryType boundaryCons[D] = {
          lsDomain<NumericType, D>::BoundaryType::REFLECTIVE_BOUNDARY,
          lsDomain<NumericType, D>::BoundaryType::INFINITE_BOUNDARY};
      if (periodicBoundary) {
        boundaryCons[0] =
            lsDomain<NumericType, D>::BoundaryType::PERIODIC_BOUNDARY;
      }

      auto substrate = LSPtrType::New(bounds, boundaryCons, gridDelta);
      NumericType normal[D] = {0., 1.};
      NumericType origin[D] = {0., baseHeight};
      lsMakeGeometry<NumericType, D>(
          substrate,
          lsSmartPointer<lsPlane<NumericType, D>>::New(origin, normal))
          .apply();

      auto mask = LSPtrType::New(bounds, boundaryCons, gridDelta);

      if (taperAngle == 0.) {
        NumericType minPoint[D] = {-finWidth / 2, baseHeight - gridDelta};
        NumericType maxPoint[D] = {finWidth / 2, baseHeight + finHeight};
        lsMakeGeometry<NumericType, D>(
            mask,
            lsSmartPointer<lsBox<NumericType, D>>::New(minPoint, maxPoint))
            .apply();
      } else {
        if (taperAngle >= 90 || taperAngle <= -90) {
          psLogger::getInstance()
              .addError("psMakeFin: Taper angle must be between -90 and 90 "
                        "degrees!")
              .print();
          return;
        }

        auto boxMesh = psSmartPointer<lsMesh<NumericType>>::New();
        boxMesh->insertNextNode({-finWidth / 2, baseHeight - gridDelta});
        boxMesh->insertNextNode({finWidth / 2, baseHeight - gridDelta});
        boxMesh->insertNextLine({1, 0});

        NumericType taperAngleRad = taperAngle * M_PI / 180.;
        NumericType offSet = finHeight * std::tan(taperAngleRad);
        if (offSet >= finWidth / 2) {
          boxMesh->insertNextNode(
              {0., baseHeight + finWidth / 2 / std::tan(taperAngleRad)});
          boxMesh->insertNextLine({2, 1});
          boxMesh->insertNextLine({0, 2});
        } else {
          boxMesh->insertNextNode(
              {finWidth / 2 - offSet, baseHeight + finHeight});
          boxMesh->insertNextNode(
              {-finWidth / 2 + offSet, baseHeight + finHeight});
          boxMesh->insertNextLine({2, 1});
          boxMesh->insertNextLine({3, 2});
          boxMesh->insertNextLine({0, 3});
        }

        lsFromSurfaceMesh<NumericType, D>(mask, boxMesh).apply();
      }

      lsBooleanOperation<NumericType, D>(substrate, mask,
                                         lsBooleanOperationEnum::UNION)
          .apply();

      if (material == psMaterial::None) {
        if (makeMask)
          domain->insertNextLevelSet(mask);
        domain->insertNextLevelSet(substrate, false);
      } else {
        if (makeMask)
          domain->insertNextLevelSetAsMaterial(mask, psMaterial::Mask);
        domain->insertNextLevelSetAsMaterial(substrate, material, false);
      }
    }
  }
};