#pragma once

#include "../geometries/psGeometryBase.hpp"
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
template <class NumericType, int D>
class MakeHole : public GeometryBase<NumericType, D> {
  using typename GeometryBase<NumericType, D>::lsDomainType;
  using typename GeometryBase<NumericType, D>::psDomainType;
  using GeometryBase<NumericType, D>::domain_;

  const NumericType holeRadius_;
  const NumericType holeDepth_;
  const NumericType holeTaperAngle_; // angle in degrees

  const NumericType maskHeight_;
  const NumericType maskTaperAngle_;

  const NumericType base_;
  const Material material_;

  HoleShape shape_;

public:
  MakeHole(psDomainType domain, NumericType holeRadius, NumericType holeDepth,
           NumericType holeTaperAngle = 0., NumericType maskHeight = 0.,
           NumericType maskTaperAngle = 0., HoleShape shape = HoleShape::Full,
           Material material = Material::Si)
      : GeometryBase<NumericType, D>(domain), holeRadius_(holeRadius),
        holeDepth_(holeDepth), holeTaperAngle_(holeTaperAngle),
        maskHeight_(maskHeight), maskTaperAngle_(maskTaperAngle), base_(0.0),
        material_(material), shape_(shape) {}

  MakeHole(psDomainType domain, NumericType gridDelta, NumericType xExtent,
           NumericType yExtent, NumericType holeRadius, NumericType holeDepth,
           NumericType taperAngle = 0., NumericType baseHeight = 0.,
           bool periodicBoundary = false, bool makeMask = false,
           Material material = Material::Si, HoleShape shape = HoleShape::Full)
      : GeometryBase<NumericType, D>(domain), holeRadius_(holeRadius),
        holeDepth_(makeMask ? 0 : holeDepth),
        holeTaperAngle_(makeMask ? 0 : taperAngle),
        maskHeight_(makeMask ? holeDepth : 0),
        maskTaperAngle_(makeMask ? taperAngle : 0), base_(baseHeight),
        material_(material), shape_(shape) {
    domain_->setup(gridDelta, xExtent, yExtent, periodicBoundary);
  }

  void apply() {
    if constexpr (D != 3) {
      Logger::getInstance()
          .addWarning("MakeHole: Hole geometry can only be created in 3D! "
                      "Falling back to trench geometry.")
          .print();
      bool halfTrench =
          shape_ == HoleShape::Half || shape_ == HoleShape::Quarter;
      MakeTrench<NumericType, D>(domain_, 2 * holeRadius_, holeDepth_,
                                 holeTaperAngle_, maskHeight_, maskTaperAngle_,
                                 halfTrench, material_)
          .apply();
      return;
    }

    auto &setup = domain_->getSetup();
    auto bounds = setup.bounds_;
    auto boundaryCons = setup.boundaryCons_;
    auto gridDelta = setup.gridDelta_;
    if (gridDelta == 0.0) {
      Logger::getInstance()
          .addWarning("MakeHole: Domain setup is not initialized.")
          .print();
      return;
    }

    if (setup.hasPeriodicBoundary() &&
        (shape_ == HoleShape::Half || shape_ == HoleShape::Quarter)) {
      Logger::getInstance()
          .addWarning("MakeHole: 'Half' or 'Quarter' shapes do not support "
                      "periodic boundaries! Creating full hole.")
          .print();
      shape_ = HoleShape::Full;
    }

    if (shape_ == HoleShape::Half) {
      setup.bounds_[0] = 0.0;
    } else if (shape_ == HoleShape::Quarter) {
      setup.bounds_[0] = 0.0;
      setup.bounds_[2] = 0.0;
    }

    domain_->clear(); // does not clear setup

    auto substrate = this->makeSubstrate(base_);

    if (maskHeight_ > 0.) {
      auto mask = this->makeMask(base_, maskHeight_);

      auto cutout = lsDomainType::New(bounds, boundaryCons, gridDelta);
      NumericType radius = holeRadius_;
      if (holeTaperAngle_ > 0. && holeDepth_ > 0.) {
        radius +=
            std::tan(holeTaperAngle_ * M_PI / 180.) * (holeDepth_ + gridDelta);
      }
      getCutout(cutout, radius, maskHeight_, base_, maskTaperAngle_);

      viennals::BooleanOperation<NumericType, D>(
          mask, cutout, viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT)
          .apply();

      domain_->insertNextLevelSetAsMaterial(mask, Material::Mask);
    }

    if (holeDepth_ > 0.) {
      auto cutout = lsDomainType::New(bounds, boundaryCons, gridDelta);
      getCutout(cutout, holeRadius_, holeDepth_ + gridDelta, base_ - holeDepth_,
                holeTaperAngle_);
      viennals::BooleanOperation<NumericType, D>(
          substrate, cutout,
          viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT)
          .apply();
    }

    domain_->insertNextLevelSetAsMaterial(substrate, material_);
  }

private:
  void getCutout(lsDomainType cutout, NumericType radius, NumericType height,
                 NumericType base, NumericType angle) {

    NumericType origin[D] = {0.};
    NumericType normal[D] = {0.};
    normal[D - 1] = 1.;
    origin[D - 1] = base;

    NumericType topRadius = radius;
    if (angle > 0.) {
      topRadius += std::tan(angle * M_PI / 180.) * height;
    }

    viennals::MakeGeometry<NumericType, D>(
        cutout, SmartPointer<viennals::Cylinder<NumericType, D>>::New(
                    origin, normal, height, radius, topRadius))
        .apply();
  }
};

} // namespace viennaps
