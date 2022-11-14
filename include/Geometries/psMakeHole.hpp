#pragma once

#include <lsBooleanOperation.hpp>
#include <lsDomain.hpp>
#include <lsMakeGeometry.hpp>

#include <psDomain.hpp>
#include <psMakeTrench.hpp>

/**
  Creates a hole geometry in z direction. For 2D geometries a regular trench is
  created.
*/
template <class NumericType, int D> class psMakeHole {
  using LSPtrType = psSmartPointer<lsDomain<NumericType, D>>;
  using PSPtrType = psSmartPointer<psDomain<NumericType, D>>;

public:
  PSPtrType domain = nullptr;

  NumericType gridDelta = .25;
  NumericType xExtent = 10;
  NumericType yExtent = 10;

  NumericType holeRadius = 3;
  NumericType taperAngle = 0; // tapering angle in degrees
  NumericType holeDepth = 6;
  bool makeMask = true;

  psMakeHole(PSPtrType passedDomain) : domain(passedDomain) {}

  psMakeHole(PSPtrType passedDomain, const NumericType passedGridDelta,
             const NumericType passedXExtent, const NumericType passedYExtent,
             const NumericType passedHoleRadius,
             const NumericType passedHoleDepth,
             const bool passedMakeMask = true)
      : domain(passedDomain), gridDelta(passedGridDelta),
        xExtent(passedXExtent), yExtent(passedYExtent),
        holeRadius(passedHoleRadius), holeDepth(passedHoleDepth),
        makeMask(passedMakeMask) {}

  psMakeHole(PSPtrType passedDomain, const NumericType passedGridDelta,
             const NumericType passedXExtent, const NumericType passedYExtent,
             const NumericType passedHoleRadius,
             const NumericType passedHoleDepth,
             const NumericType passedTaperAngle,
             const bool passedMakeMask = true)
      : domain(passedDomain), gridDelta(passedGridDelta),
        xExtent(passedXExtent), yExtent(passedYExtent),
        holeRadius(passedHoleRadius), taperAngle(passedTaperAngle),
        holeDepth(passedHoleDepth), makeMask(passedMakeMask) {}

  void apply() {
    if (D != 3) {
      lsMessage::getInstance()
          .addWarning("psMakeHole: Hole geometry can only be created in 3D! "
                      "Falling back to trench geometry.")
          .print();
      psMakeTrench<NumericType, D>(domain, gridDelta, xExtent, yExtent,
                                   2 * holeRadius, taperAngle, holeDepth)
          .apply();

      return;
    }
    domain->clear();
    double bounds[2 * D];
    bounds[0] = -xExtent / 2.;
    bounds[1] = xExtent / 2.;

    if constexpr (D == 3) {
      bounds[2] = -yExtent / 2.;
      bounds[3] = yExtent / 2.;
      bounds[4] = -gridDelta;
      bounds[5] = holeDepth + gridDelta;
    } else {
      bounds[2] = -gridDelta;
      bounds[3] = holeDepth + gridDelta;
    }

    typename lsDomain<NumericType, D>::BoundaryType boundaryCons[D];

    for (int i = 0; i < D - 1; i++)
      boundaryCons[i] =
          lsDomain<NumericType, D>::BoundaryType::REFLECTIVE_BOUNDARY;
    boundaryCons[D - 1] =
        lsDomain<NumericType, D>::BoundaryType::INFINITE_BOUNDARY;

    // substrate
    auto substrate = LSPtrType::New(bounds, boundaryCons, gridDelta);
    NumericType normal[D] = {0.};
    NumericType origin[D] = {0.};
    normal[D - 1] = 1.;
    lsMakeGeometry<NumericType, D>(
        substrate, lsSmartPointer<lsPlane<NumericType, D>>::New(origin, normal))
        .apply();

    // mask layer
    auto mask = LSPtrType::New(bounds, boundaryCons, gridDelta);
    origin[D - 1] = holeDepth;
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

    // cylinder cutout
    normal[D - 1] = 1.;
    origin[D - 1] = 0.;

    NumericType topRadius = holeRadius;
    if (taperAngle) {
      topRadius += std::tan(taperAngle * rayInternal::PI / 180.) * holeDepth;
    }

    lsMakeGeometry<NumericType, D>(
        maskAdd,
        lsSmartPointer<lsCylinder<NumericType, D>>::New(
            origin, normal, holeDepth + 2 * gridDelta, holeRadius, topRadius))
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
