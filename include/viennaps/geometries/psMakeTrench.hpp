#pragma once

#include "../psDomain.hpp"
#include "psGeometryFactory.hpp"

#include <lsBooleanOperation.hpp>
#include <lsFromSurfaceMesh.hpp>
#include <lsMakeGeometry.hpp>

namespace viennaps {

using namespace viennacore;

/// Generates new a trench geometry extending in the z (3D) or y (2D) direction,
/// centrally positioned at the origin with the total extent specified in the x
/// and y directions. The trench configuration may include periodic boundaries
/// in both the x and y directions. Users have the flexibility to define the
/// trench's width, depth, and incorporate tapering with a designated angle.
/// Moreover, the trench can serve as a mask, applying the specified material_
/// exclusively to the bottom while the remaining portion adopts the mask
/// material_.
template <class NumericType, int D> class MakeTrench {
  using psDomainType = SmartPointer<Domain<NumericType, D>>;

  psDomainType domain_;
  GeometryFactory<NumericType, D> geometryFactory_;
  static constexpr NumericType eps_ = 1e-4;

  const NumericType trenchWidth_;
  const NumericType trenchDepth_;
  const NumericType trenchTaperAngle_; // angle in degrees

  const NumericType maskHeight_;
  const NumericType maskTaperAngle_;

  const NumericType base_;
  const Material material_;
  const Material maskMaterial_ = Material::Mask;

public:
  MakeTrench(psDomainType domain, NumericType trenchWidth,
             NumericType trenchDepth, NumericType trenchTaperAngle = 0,
             NumericType maskHeight = 0, NumericType maskTaperAngle = 0,
             bool halfTrench = false, Material material = Material::Si,
             Material maskMaterial = Material::Mask)
      : domain_(domain), geometryFactory_(domain->getSetup(), __func__),
        trenchWidth_(trenchWidth), trenchDepth_(trenchDepth),
        trenchTaperAngle_(trenchTaperAngle), maskHeight_(maskHeight),
        maskTaperAngle_(maskTaperAngle), base_(0.0), material_(material),
        maskMaterial_(maskMaterial) {
    if (halfTrench)
      domain_->getSetup().halveXAxis();
  }

  MakeTrench(psDomainType domain, NumericType gridDelta, NumericType xExtent,
             NumericType yExtent, NumericType trenchWidth,
             NumericType trenchDepth, NumericType taperAngle = 0.,
             NumericType base = 0., bool periodicBoundary = false,
             bool makeMask = false, Material material = Material::Si)
      : domain_(domain), geometryFactory_(domain->getSetup(), __func__),
        trenchWidth_(trenchWidth), trenchDepth_(makeMask ? 0 : trenchDepth),
        trenchTaperAngle_(makeMask ? 0 : taperAngle),
        maskHeight_(makeMask ? trenchDepth : 0),
        maskTaperAngle_(makeMask ? taperAngle : 0), base_(base),
        material_(material) {
    domain_->setup(gridDelta, xExtent, yExtent,
                   periodicBoundary ? BoundaryType::PERIODIC_BOUNDARY
                                    : BoundaryType::REFLECTIVE_BOUNDARY);
  }

  void apply() {
    domain_->clear(); // this does not clear the setup
    domain_->getSetup().check();

    if (maskHeight_ > 0.) {
      auto mask = geometryFactory_.makeMask(base_ - eps_, maskHeight_);
      domain_->insertNextLevelSetAsMaterial(mask, maskMaterial_);
      std::array<NumericType, D> position = {0.};
      position[D - 1] = base_ - 2 * eps_;
      auto cutout = geometryFactory_.makeBoxStencil(
          position, trenchWidth_, maskHeight_ + 3 * eps_, -maskTaperAngle_);
      domain_->applyBooleanOperation(
          cutout, viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT);
    }

    auto substrate = geometryFactory_.makeSubstrate(base_);
    domain_->insertNextLevelSetAsMaterial(substrate, material_);

    if (trenchDepth_ > 0.) {
      std::array<NumericType, D> position = {0.};
      position[D - 1] = base_;
      auto cutout = geometryFactory_.makeBoxStencil(
          position, trenchWidth_, -trenchDepth_, -trenchTaperAngle_);
      domain_->applyBooleanOperation(cutout,
                                     viennals::BooleanOperationEnum::INTERSECT);
    }
  }
};

} // namespace viennaps
