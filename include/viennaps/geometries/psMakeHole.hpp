#pragma once

#include "../psDomain.hpp"
#include "psGeometryFactory.hpp"
#include "psMakeTrench.hpp"

#include <lsBooleanOperation.hpp>

namespace viennaps {

enum class HoleShape { FULL, HALF, QUARTER };

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
  using psDomainType = SmartPointer<Domain<NumericType, D>>;

  psDomainType domain_;
  GeometryFactory<NumericType, D> geometryFactory_;
  static constexpr NumericType eps_ = 1e-4;

  const NumericType holeRadius_;
  const NumericType holeDepth_;
  const NumericType holeTaperAngle_; // angle in degrees

  const NumericType maskHeight_;
  const NumericType maskTaperAngle_;

  const NumericType base_;
  const Material material_;
  const Material maskMaterial_ = Material::Mask;

  HoleShape shape_;

public:
  MakeHole(psDomainType domain, NumericType holeRadius, NumericType holeDepth,
           NumericType holeTaperAngle = 0., NumericType maskHeight = 0.,
           NumericType maskTaperAngle = 0., HoleShape shape = HoleShape::FULL,
           Material material = Material::Si,
           Material maskMaterial = Material::Mask)
      : domain_(domain), geometryFactory_(domain->getSetup(), __func__),
        holeRadius_(holeRadius), holeDepth_(holeDepth),
        holeTaperAngle_(holeTaperAngle), maskHeight_(maskHeight),
        maskTaperAngle_(maskTaperAngle), base_(0.0), material_(material),
        maskMaterial_(maskMaterial), shape_(shape) {}

  MakeHole(psDomainType domain, NumericType gridDelta, NumericType xExtent,
           NumericType yExtent, NumericType holeRadius, NumericType holeDepth,
           NumericType taperAngle = 0., NumericType baseHeight = 0.,
           bool periodicBoundary = false, bool makeMask = false,
           Material material = Material::Si, HoleShape shape = HoleShape::FULL)
      : domain_(domain), geometryFactory_(domain->getSetup(), __func__),
        holeRadius_(holeRadius), holeDepth_(makeMask ? 0 : holeDepth),
        holeTaperAngle_(makeMask ? 0 : taperAngle),
        maskHeight_(makeMask ? holeDepth : 0),
        maskTaperAngle_(makeMask ? taperAngle : 0), base_(baseHeight),
        material_(material), shape_(shape) {
    domain_->setup(gridDelta, xExtent, yExtent,
                   periodicBoundary ? BoundaryType::PERIODIC_BOUNDARY
                                    : BoundaryType::REFLECTIVE_BOUNDARY);
  }

  void apply() {
    if constexpr (D != 3) {
      Logger::getInstance()
          .addWarning("MakeHole: Hole geometry can only be created in 3D! "
                      "Using trench geometry instead.")
          .print();
      bool halfTrench =
          shape_ == HoleShape::HALF || shape_ == HoleShape::QUARTER;
      MakeTrench<NumericType, D>(domain_, 2 * holeRadius_, holeDepth_,
                                 holeTaperAngle_, maskHeight_, maskTaperAngle_,
                                 halfTrench, material_)
          .apply();
      return;
    }

    domain_->clear(); // this does not clear the setup
    domain_->getSetup().check();

    auto &setup = domain_->getSetup();

    if (setup.hasPeriodicBoundary() &&
        (shape_ == HoleShape::HALF || shape_ == HoleShape::QUARTER)) {
      Logger::getInstance()
          .addWarning("MakeHole: 'HALF' or 'QUARTER' shapes do not support "
                      "periodic boundaries! Creating full hole.")
          .print();
      shape_ = HoleShape::FULL;
    }

    if (shape_ == HoleShape::HALF) {
      setup.halveXAxis();
    } else if (shape_ == HoleShape::QUARTER) {
      setup.halveXAxis();
      setup.halveYAxis();
    }

    auto substrate = geometryFactory_.makeSubstrate(base_);

    if (maskHeight_ > 0.) {
      auto mask = geometryFactory_.makeMask(base_ - eps_, maskHeight_);
      domain_->insertNextLevelSetAsMaterial(mask, maskMaterial_);
      std::array<NumericType, D> position = {0.};
      position[D - 1] = base_ - 2 * eps_;
      auto cutout = geometryFactory_.makeCylinderStencil(
          position, holeRadius_, maskHeight_ + 3 * eps_, maskTaperAngle_);
      domain_->applyBooleanOperation(
          cutout, viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT);
    }

    domain_->insertNextLevelSetAsMaterial(substrate, material_);

    if (holeDepth_ > 0.) {
      std::array<NumericType, D> position = {0.};
      position[D - 1] = base_;
      auto cutout = geometryFactory_.makeCylinderStencil(
          position, holeRadius_, -holeDepth_, holeTaperAngle_);
      domain_->applyBooleanOperation(cutout,
                                     viennals::BooleanOperationEnum::INTERSECT);
    }
  }
};

PS_PRECOMPILE_PRECISION_DIMENSION(MakeHole)

} // namespace viennaps
