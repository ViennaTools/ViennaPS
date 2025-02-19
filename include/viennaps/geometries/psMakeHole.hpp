#pragma once

#include "../geometries/psMakeTrench.hpp"
#include "../psDomain.hpp"

#include <lsBooleanOperation.hpp>
#include <lsMakeGeometry.hpp>

namespace viennaps {

enum class HoleShape { Full, Half, Quarter };

using namespace viennacore;

/// Generates new a hole geometry in the z direction, which, in 2D mode,
/// corresponds to a trench geometry. Positioned at the origin, the hole is
/// centered, with the total extent defined in the x and y directions. The
/// normal direction for the hole creation is in the positive z direction in 3D
/// and the positive y direction in 2D. Users can specify the hole's radius,
/// depth, and opt for tapering with a designated angle. The hole configuration
/// may include periodic boundaries in both the x and y directions.
/// Additionally, the hole can serve as a mask, with the specified material only
/// applied to the bottom of the hole, while the remainder adopts the mask
/// material.
template <class NumericType, int D> class MakeHole {
  using lsDomainType = SmartPointer<viennals::Domain<NumericType, D>>;
  using psDomainType = SmartPointer<Domain<NumericType, D>>;
  using BoundaryEnum = typename viennals::Domain<NumericType, D>::BoundaryType;

  psDomainType domain_ = nullptr;

  const NumericType gridDelta_;
  const NumericType xExtent_;
  const NumericType yExtent_;

  const NumericType holeRadius_;
  const NumericType holeDepth_;
  const NumericType taperAngle_; // taper angle in degrees
  const NumericType baseHeight_;

  const bool makeMask_;
  const bool periodicBoundary_;
  const Material material_;

  const HoleShape shape_;

public:
  MakeHole(psDomainType domain, NumericType gridDelta, NumericType xExtent,
           NumericType yExtent, NumericType holeRadius, NumericType holeDepth,
           NumericType taperAngle = 0., NumericType baseHeight = 0.,
           bool periodicBoundary = false, bool makeMask = false,
           Material material = Material::Undefined,
           HoleShape shape = HoleShape::Full)
      : domain_(domain), gridDelta_(gridDelta), xExtent_(xExtent),
        yExtent_(yExtent), holeRadius_(holeRadius), holeDepth_(holeDepth),
        shape_(shape), taperAngle_(taperAngle), baseHeight_(baseHeight),
        periodicBoundary_(periodicBoundary && (shape != HoleShape::Half &&
                                               shape != HoleShape::Quarter)),
        makeMask_(makeMask), material_(material) {
    if (periodicBoundary &&
        (shape == HoleShape::Half || shape == HoleShape::Quarter)) {
      Logger::getInstance()
          .addWarning("MakeHole: 'Half' or 'Quarter' shapes do not support "
                      "periodic boundaries! "
                      "Defaulting to reflective boundaries!")
          .print();
    }
  }

  void apply() {

    if constexpr (D != 3) {
      Logger::getInstance()
          .addWarning("MakeHole: Hole geometry can only be created in 3D! "
                      "Falling back to trench geometry.")
          .print();
      MakeTrench<NumericType, D>(
          domain_, gridDelta_, xExtent_, yExtent_, 2 * holeRadius_, holeDepth_,
          taperAngle_, baseHeight_, periodicBoundary_, makeMask_, material_)
          .apply();

      return;
    }

    domain_->clear();
    double bounds[2 * D];
    bounds[0] = (shape_ != HoleShape::Full) ? 0. : -xExtent_ / 2.;
    bounds[1] = xExtent_ / 2.;

    if constexpr (D == 3) {
      bounds[2] = (shape_ == HoleShape::Quarter) ? 0. : -yExtent_ / 2.;
      bounds[3] = yExtent_ / 2.;
      bounds[4] = baseHeight_ - gridDelta_;
      bounds[5] = baseHeight_ + holeDepth_ + gridDelta_;
    } else {
      bounds[2] = baseHeight_ - gridDelta_;
      bounds[3] = baseHeight_ + holeDepth_ + gridDelta_;
    }

    BoundaryEnum boundaryCons[D];
    for (int i = 0; i < D - 1; i++) {
      if (periodicBoundary_) {
        boundaryCons[i] = BoundaryEnum::PERIODIC_BOUNDARY;
      } else {
        boundaryCons[i] = BoundaryEnum::REFLECTIVE_BOUNDARY;
      }
    }
    boundaryCons[D - 1] = BoundaryEnum::INFINITE_BOUNDARY;

    // substrate
    auto substrate = lsDomainType::New(bounds, boundaryCons, gridDelta_);
    NumericType normal[D] = {0.};
    NumericType origin[D] = {0.};
    normal[D - 1] = 1.;
    origin[D - 1] = baseHeight_;
    viennals::MakeGeometry<NumericType, D>(
        substrate,
        SmartPointer<viennals::Plane<NumericType, D>>::New(origin, normal))
        .apply();

    // mask layer
    auto mask = lsDomainType::New(bounds, boundaryCons, gridDelta_);
    origin[D - 1] = holeDepth_ + baseHeight_;
    viennals::MakeGeometry<NumericType, D>(
        mask,
        SmartPointer<viennals::Plane<NumericType, D>>::New(origin, normal))
        .apply();

    auto maskAdd = lsDomainType::New(bounds, boundaryCons, gridDelta_);
    origin[D - 1] = baseHeight_;
    normal[D - 1] = -1.;
    viennals::MakeGeometry<NumericType, D>(
        maskAdd,
        SmartPointer<viennals::Plane<NumericType, D>>::New(origin, normal))
        .apply();

    viennals::BooleanOperation<NumericType, D>(
        mask, maskAdd, viennals::BooleanOperationEnum::INTERSECT)
        .apply();

    // cylinder cutout
    normal[D - 1] = 1.;
    origin[D - 1] = baseHeight_;

    NumericType topRadius = holeRadius_;
    if (taperAngle_) {
      topRadius += std::tan(taperAngle_ * M_PI / 180.) * holeDepth_;
    }

    viennals::MakeGeometry<NumericType, D>(
        maskAdd, SmartPointer<viennals::Cylinder<NumericType, D>>::New(
                     origin, normal, holeDepth_ + 2 * gridDelta_, holeRadius_,
                     topRadius))
        .apply();

    viennals::BooleanOperation<NumericType, D>(
        mask, maskAdd, viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT)
        .apply();

    viennals::BooleanOperation<NumericType, D>(
        substrate, mask, viennals::BooleanOperationEnum::UNION)
        .apply();

    if (material_ == Material::Undefined) {
      if (makeMask_)
        domain_->insertNextLevelSet(mask);
      domain_->insertNextLevelSet(substrate, false);
    } else {
      if (makeMask_)
        domain_->insertNextLevelSetAsMaterial(mask, Material::Mask);
      domain_->insertNextLevelSetAsMaterial(substrate, material_, false);
    }
  }
};

} // namespace viennaps
