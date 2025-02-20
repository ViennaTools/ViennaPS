#pragma once

#include "../psDomain.hpp"
#include "psGeometryFactory.hpp"

#include <lsBooleanOperation.hpp>
#include <lsMakeGeometry.hpp>

namespace viennaps {

using namespace viennacore;

/// Generates a stack of alternating SiO2/Si3N4 layers featuring an optionally
/// etched hole (3D) or trench (2D) at the center. The stack emerges in the
/// positive z direction (3D) or y direction (2D) and is centered around the
/// origin, with its x/y extent specified. Users have the flexibility to
/// introduce periodic boundaries in the x and y directions. Additionally, the
/// stack can incorporate a top mask with a central hole of a specified radius
/// or a trench with a designated width. This versatile functionality enables
/// users to create diverse and customized structures for simulation scenarios.
template <class NumericType, int D> class MakeStack {
  using psDomainType = SmartPointer<Domain<NumericType, D>>;

  psDomainType domain_;
  GeometryFactory<NumericType, D> geometryFactory_;
  static constexpr NumericType eps_ = 1e-4;

  const int numLayers_;
  const NumericType layerHeight_;
  const NumericType substrateHeight_;

  NumericType holeRadius_;
  const NumericType trenchWidth_;
  const NumericType maskHeight_;
  const NumericType taperAngle_ = 0.;
  const Material maskMaterial_ = Material::Mask;

public:
  MakeStack(psDomainType domain, int numLayers, NumericType layerHeight,
            NumericType substrateHeight, NumericType holeRadius,
            NumericType trenchWidth, NumericType maskHeight,
            NumericType taperAngle, bool halfStack = false,
            Material maskMaterial = Material::Mask)
      : domain_(domain), geometryFactory_(domain->getSetup(), __func__),
        numLayers_(numLayers), layerHeight_(layerHeight),
        substrateHeight_(substrateHeight), holeRadius_(holeRadius),
        trenchWidth_(trenchWidth), maskHeight_(maskHeight),
        taperAngle_(taperAngle), maskMaterial_(maskMaterial) {
    if (halfStack)
      domain_->getSetup().halveXAxis();
  }

  MakeStack(psDomainType domain, NumericType gridDelta, NumericType xExtent,
            NumericType yExtent, int numLayers, NumericType layerHeight,
            NumericType substrateHeight, NumericType holeRadius,
            NumericType trenchWidth, NumericType maskHeight,
            bool periodicBoundary = false)
      : domain_(domain), geometryFactory_(domain->getSetup(), __func__),
        numLayers_(numLayers), layerHeight_(layerHeight),
        substrateHeight_(substrateHeight), holeRadius_(holeRadius),
        trenchWidth_(trenchWidth), maskHeight_(maskHeight) {
    domain_->setup(gridDelta, xExtent, yExtent,
                   periodicBoundary ? BoundaryType::PERIODIC_BOUNDARY
                                    : BoundaryType::REFLECTIVE_BOUNDARY);
  }

  void apply() {
    domain_->clear();
    domain_->getSetup().check();

    if (maskHeight_ > 0.) {
      NumericType maskBase = substrateHeight_ + layerHeight_ * numLayers_;
      auto mask = geometryFactory_.makeMask(maskBase - eps_, maskHeight_);
      domain_->insertNextLevelSetAsMaterial(mask, maskMaterial_);

      std::array<NumericType, D> position = {0.};
      position[D - 1] = maskBase - 2 * eps_;

      if (holeRadius_ > 0. && D == 3) {
        auto cutout = geometryFactory_.makeCylinderStencil(
            position, holeRadius_, maskHeight_ + 3 * eps_, -taperAngle_);
        domain_->applyBooleanOperation(
            cutout, viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT);
      } else {
        NumericType trenchWidth = trenchWidth_;
        if (trenchWidth == 0.) {
          trenchWidth = 2 * holeRadius_;
        }
        if (trenchWidth == 0.) {
          Logger::getInstance()
              .addError(
                  "MakeStack: Trench width or hole radius must be greater "
                  "0 to create mask.")
              .print();
        }
        auto cutout = geometryFactory_.makeBoxStencil(
            position, trenchWidth, maskHeight_ + 3 * eps_, -taperAngle_);
        domain_->applyBooleanOperation(
            cutout, viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT);
      }
    }

    // Silicon substrate
    auto substrate = geometryFactory_.makeSubstrate(substrateHeight_);
    domain_->insertNextLevelSetAsMaterial(substrate, Material::Si);

    // Si3N4/SiO2 layers
    NumericType current = substrateHeight_ + layerHeight_;
    for (int i = 0; i < numLayers_; ++i) {
      auto ls = geometryFactory_.makeSubstrate(current);
      if (i % 2 == 0) {
        domain_->insertNextLevelSetAsMaterial(ls, Material::SiO2);
      } else {
        domain_->insertNextLevelSetAsMaterial(ls, Material::Si3N4);
      }
      current += layerHeight_;
    }

    if ((holeRadius_ > 0. || trenchWidth_ > 0.) && maskHeight_ == 0.) {
      std::array<NumericType, D> position = {0.};
      position[D - 1] = substrateHeight_;

      if (holeRadius_ > 0. && D == 3) {
        auto cutout = geometryFactory_.makeCylinderStencil(
            position, holeRadius_, numLayers_ * layerHeight_ + eps_,
            -taperAngle_);
        domain_->applyBooleanOperation(
            cutout, viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT);
      } else {
        NumericType trenchWidth = trenchWidth_;
        if (trenchWidth == 0.) {
          trenchWidth = 2 * holeRadius_;
        }
        auto cutout = geometryFactory_.makeBoxStencil(
            position, trenchWidth, numLayers_ * layerHeight_ + eps_,
            -taperAngle_);
        domain_->applyBooleanOperation(
            cutout, viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT);
      }
    }
  }

  int getTopLayer() const { return numLayers_; }

  NumericType getHeight() const {
    return substrateHeight_ + numLayers_ * layerHeight_ + maskHeight_;
  }
};

} // namespace viennaps
