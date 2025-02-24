#pragma once

#include "../psDomain.hpp"
#include "psGeometryFactory.hpp"

#include <lsBooleanOperation.hpp>
#include <lsFromSurfaceMesh.hpp>
#include <lsMakeGeometry.hpp>

#include <vcLogger.hpp>

namespace viennaps {

using namespace viennacore;

/// Generates a new fin geometry extending in the z (3D) or y (2D) direction,
/// centered at the origin with specified dimensions in the x and y directions.
/// The fin may incorporate periodic boundaries in the x and y directions. Users
/// can define the width and height of the fin, and it can function as a mask,
/// with the specified material exclusively applied to the bottom of the fin,
/// while the upper portion adopts the mask material.
template <class NumericType, int D> class MakeFin {
  using psDomainType = SmartPointer<Domain<NumericType, D>>;

  psDomainType domain_;
  GeometryFactory<NumericType, D> geometryFactory_;
  static constexpr NumericType eps_ = 1e-4;

  const NumericType finWidth_;
  const NumericType finHeight_;
  const NumericType finTaperAngle_; // taper angle in degrees

  const NumericType maskHeight_;
  const NumericType maskTaperAngle_;

  const NumericType base_;
  const Material material_;
  const Material maskMaterial_ = Material::Mask;

public:
  MakeFin(psDomainType domain, NumericType finWidth, NumericType finHeight,
          NumericType finTaperAngle = 0., NumericType maskHeight = 0.,
          NumericType maskTaperAngle = 0., bool halfFin = false,
          Material material = Material::Si,
          Material maskMaterial = Material::Mask)
      : domain_(domain), geometryFactory_(domain->getSetup(), __func__),
        finWidth_(finWidth), finHeight_(finHeight),
        finTaperAngle_(finTaperAngle), maskHeight_(maskHeight),
        maskTaperAngle_(maskTaperAngle), base_(0.0), material_(material),
        maskMaterial_(maskMaterial) {
    if (halfFin)
      domain_->getSetup().halveXAxis();
  }

  MakeFin(psDomainType domain, NumericType gridDelta, NumericType xExtent,
          NumericType yExtent, NumericType finWidth, NumericType finHeight,
          NumericType taperAngle = 0., NumericType baseHeight = 0.,
          bool periodicBoundary = false, bool makeMask = false,
          Material material = Material::Si)
      : domain_(domain), geometryFactory_(domain->getSetup(), __func__),
        finWidth_(finWidth), finHeight_(makeMask ? 0 : finHeight),
        finTaperAngle_(makeMask ? 0 : taperAngle),
        maskHeight_(makeMask ? finHeight : 0),
        maskTaperAngle_(makeMask ? taperAngle : 0), base_(baseHeight),
        material_(material) {
    domain_->setup(gridDelta, xExtent, yExtent,
                   periodicBoundary ? BoundaryType::PERIODIC_BOUNDARY
                                    : BoundaryType::REFLECTIVE_BOUNDARY);
  }

  void apply() {
    domain_->clear(); // this does not clear the setup
    domain_->getSetup().check();

    if (maskHeight_ > 0.) {
      auto position = std::array<NumericType, D>{0.};
      position[D - 1] = base_ + finHeight_ - eps_;
      auto mask = geometryFactory_.makeBoxStencil(position, finWidth_,
                                                  maskHeight_, maskTaperAngle_);
      domain_->insertNextLevelSetAsMaterial(mask, maskMaterial_);
    }

    auto substrate = geometryFactory_.makeSubstrate(base_);

    if (finHeight_ > 0.) {
      auto position = std::array<NumericType, D>{0.};
      position[D - 1] = base_ + finHeight_;
      auto fin = geometryFactory_.makeBoxStencil(
          position, finWidth_, -finHeight_ - eps_, finTaperAngle_);
      viennals::BooleanOperation<NumericType, D>(
          fin, viennals::BooleanOperationEnum::INVERT)
          .apply();
      viennals::BooleanOperation<NumericType, D>(
          substrate, fin, viennals::BooleanOperationEnum::UNION)
          .apply();
    }

    domain_->insertNextLevelSetAsMaterial(substrate, material_);
  }
};

} // namespace viennaps
