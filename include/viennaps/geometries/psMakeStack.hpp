#pragma once

#include "../psDomain.hpp"
#include "psGeometryBase.hpp"

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
template <class NumericType, int D>
class MakeStack : public GeometryBase<NumericType, D> {
  using typename GeometryBase<NumericType, D>::lsDomainType;
  using typename GeometryBase<NumericType, D>::psDomainType;
  using GeometryBase<NumericType, D>::domain_;
  using GeometryBase<NumericType, D>::name_;
  using GeometryBase<NumericType, D>::eps_;

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
      : GeometryBase<NumericType, D>(domain, __func__), numLayers_(numLayers),
        layerHeight_(layerHeight), substrateHeight_(substrateHeight),
        holeRadius_(holeRadius), trenchWidth_(trenchWidth),
        maskHeight_(maskHeight), taperAngle_(taperAngle),
        maskMaterial_(maskMaterial) {
    if (halfStack)
      domain_->getSetup().halveXAxis();
  }

  MakeStack(psDomainType domain, NumericType gridDelta, NumericType xExtent,
            NumericType yExtent, int numLayers, NumericType layerHeight,
            NumericType substrateHeight, NumericType holeRadius,
            NumericType trenchWidth, NumericType maskHeight,
            bool periodicBoundary = false)
      : GeometryBase<NumericType, D>(domain, __func__), numLayers_(numLayers),
        layerHeight_(layerHeight), substrateHeight_(substrateHeight),
        holeRadius_(holeRadius), trenchWidth_(trenchWidth),
        maskHeight_(maskHeight) {
    domain_->setup(gridDelta, xExtent, yExtent,
                   periodicBoundary ? BoundaryType::PERIODIC_BOUNDARY
                                    : BoundaryType::REFLECTIVE_BOUNDARY);
  }

  void apply() {
    domain_->clear();
    if (!this->setupCheck())
      return;

    if (maskHeight_ > 0.) {
      NumericType maskBase = substrateHeight_ + layerHeight_ * numLayers_;
      auto mask = this->makeMask(maskBase - eps_, maskHeight_);
      domain_->insertNextLevelSetAsMaterial(mask, maskMaterial_);

      std::array<NumericType, D> position = {0.};
      position[D - 1] = maskBase - 2 * eps_;

      if (holeRadius_ > 0. && D == 3) {
        auto cutout = this->makeCylinderStencil(
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
              .addError(name_ + ": Trench width or hole radius must be greater "
                                "0 to create mask.")
              .print();
        }
        auto cutout = this->makeTrenchStencil(
            position, trenchWidth, maskHeight_ + 3 * eps_, -taperAngle_);
        domain_->applyBooleanOperation(
            cutout, viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT);
      }
    }

    // Silicon substrate
    auto substrate = this->makeSubstrate(substrateHeight_);
    domain_->insertNextLevelSetAsMaterial(substrate, Material::Si);

    // Si3N4/SiO2 layers
    NumericType current = substrateHeight_ + layerHeight_;
    for (int i = 0; i < numLayers_; ++i) {
      auto ls = this->makeSubstrate(current);
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
        auto cutout = this->makeCylinderStencil(
            position, holeRadius_, numLayers_ * layerHeight_ + eps_,
            -taperAngle_);
        domain_->applyBooleanOperation(
            cutout, viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT);
      } else {
        NumericType trenchWidth = trenchWidth_;
        if (trenchWidth == 0.) {
          trenchWidth = 2 * holeRadius_;
        }
        auto cutout = this->makeTrenchStencil(position, trenchWidth,
                                              numLayers_ * layerHeight_ + eps_,
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

private:
  // void create2DGeometry() {
  //   auto setup = domain_->getSetup();
  //   auto bounds = setup.bounds_;
  //   auto boundaryConds = setup.boundaryCons_;
  //   auto gridDelta = setup.gridDelta_;

  //   if (maskHeight_ > 0.) {
  //     NumericType maskBase = substrateHeight_ + layerHeight_ * numLayers_;
  //     auto mask = this->makeMask(maskBase, maskHeight_);

  //     if (holeRadius_ == 0.) {
  //       holeRadius_ = trenchWidth_ / 2.;
  //     }
  //     auto cutout = lsDomainType::New(bounds, boundaryConds, gridDelta);
  //     NumericType minPoint[D] = {-holeRadius_, maskBase - gridDelta};
  //     NumericType maxPoint[D] = {holeRadius_,
  //                                maskBase + maskHeight_ + gridDelta};
  //     viennals::MakeGeometry<NumericType, D>(
  //         cutout,
  //         SmartPointer<viennals::Box<NumericType, D>>::New(minPoint,
  //         maxPoint)) .apply();

  //     viennals::BooleanOperation<NumericType, D>(
  //         mask, cutout, viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT)
  //         .apply();

  //     domain_->insertNextLevelSetAsMaterial(mask, maskMaterial_);
  //   }

  //   // Silicon substrate
  //   auto substrate = this->makeSubstrate(substrateHeight_);
  //   domain_->insertNextLevelSetAsMaterial(substrate, Material::Si);

  //   // Si3N4/SiO2 layers
  //   NumericType current = substrateHeight_ + layerHeight_;
  //   for (int i = 0; i < numLayers_; ++i) {
  //     auto ls = lsDomainType::New(bounds, boundaryConds, gridDelta);
  //     origin_[D - 1] = substrateHeight_ + layerHeight_ * (i + 1);
  //     viennals::MakeGeometry<NumericType, D>(
  //         ls,
  //         SmartPointer<viennals::Plane<NumericType, D>>::New(origin_,
  //         normal_)) .apply();
  //     if (i % 2 == 0) {
  //       domain_->insertNextLevelSetAsMaterial(ls, Material::SiO2);
  //     } else {
  //       domain_->insertNextLevelSetAsMaterial(ls, Material::Si3N4);
  //     }
  //   }

  //   if ((holeRadius_ > 0. || trenchWidth_ > 0.) && maskHeight_ == 0.) {
  //     if (holeRadius_ == 0.) {
  //       holeRadius_ = trenchWidth_ / 2.;
  //     }
  //     // cut out middle
  //     auto cutOut = lsDomainType::New(bounds, boundaryConds, gridDelta);
  //     NumericType minPoint[D] = {-holeRadius_, 0.};
  //     NumericType maxPoint[D] = {holeRadius_, substrateHeight_ +
  //                                                 layerHeight_ * numLayers_ +
  //                                                 maskHeight_ + gridDelta};
  //     viennals::MakeGeometry<NumericType, D>(
  //         cutOut,
  //         SmartPointer<viennals::Box<NumericType, D>>::New(minPoint,
  //         maxPoint)) .apply();

  //     domain_->applyBooleanOperation(
  //         cutOut, viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT);
  //   }
  // }

  // void create3DGeometry() {
  //   domain_->clear();

  //   if (maskHeight_ > 0.) {
  //     auto mask = lsDomainType::New(bounds_, boundaryConds, gridDelta);
  //     origin_[D - 1] =
  //         substrateHeight_ + layerHeight_ * numLayers_ + maskHeight_;
  //     viennals::MakeGeometry<NumericType, D>(
  //         mask,
  //         SmartPointer<viennals::Plane<NumericType, D>>::New(origin_,
  //         normal_)) .apply();

  //     auto maskAdd = lsDomainType::New(bounds_, boundaryConds_, gridDelta);
  //     origin_[D - 1] = substrateHeight_ + layerHeight_ * numLayers_;
  //     normal_[D - 1] = -1;
  //     viennals::MakeGeometry<NumericType, D>(
  //         maskAdd,
  //         SmartPointer<viennals::Plane<NumericType, D>>::New(origin_,
  //         normal_)) .apply();

  //     viennals::BooleanOperation<NumericType, D>(
  //         mask, maskAdd, viennals::BooleanOperationEnum::INTERSECT)
  //         .apply();

  //     if (holeRadius_ > 0.) {
  //       normal_[D - 1] = 1.;
  //       viennals::MakeGeometry<NumericType, D>(
  //           maskAdd,
  //           SmartPointer<viennals::Cylinder<NumericType, D>>::New(
  //               origin_, normal_, maskHeight_ + gridDelta, holeRadius_))
  //           .apply();

  //       viennals::BooleanOperation<NumericType, D>(
  //           mask, maskAdd,
  //           viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT) .apply();
  //     } else if (trenchWidth_ > 0.) {
  //       NumericType minPoint[D] = {
  //           static_cast<NumericType>(-trenchWidth_ / 2.),
  //           static_cast<NumericType>(-yExtent_ / 2. - gridDelta),
  //           origin_[D - 1]};
  //       NumericType maxPoint[D] = {
  //           static_cast<NumericType>(trenchWidth_ / 2.),
  //           static_cast<NumericType>(yExtent_ / 2. + gridDelta),
  //           origin_[D - 1] + maskHeight_ + gridDelta};
  //       viennals::MakeGeometry<NumericType, D>(
  //           maskAdd, SmartPointer<viennals::Box<NumericType,
  //           D>>::New(minPoint,
  //                                                                     maxPoint))
  //           .apply();

  //       viennals::BooleanOperation<NumericType, D>(
  //           mask, maskAdd,
  //           viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT) .apply();
  //     }

  //     domain_->insertNextLevelSetAsMaterial(mask, Material::Mask);
  //   }

  //   // Silicon substrate
  //   auto substrate = lsDomainType::New(bounds_, boundaryConds_, gridDelta);
  //   origin_[D - 1] = substrateHeight_;
  //   viennals::MakeGeometry<NumericType, D>(
  //       substrate,
  //       SmartPointer<viennals::Plane<NumericType, D>>::New(origin_, normal_))
  //       .apply();
  //   domain_->insertNextLevelSetAsMaterial(substrate, Material::Si);

  //   // Si3N4/SiO2 layers
  //   for (int i = 0; i < numLayers_; ++i) {
  //     auto ls = lsDomainType::New(bounds_, boundaryConds_, gridDelta);
  //     origin_[D - 1] = substrateHeight_ + layerHeight_ * (i + 1);
  //     viennals::MakeGeometry<NumericType, D>(
  //         ls,
  //         SmartPointer<viennals::Plane<NumericType, D>>::New(origin_,
  //         normal_)) .apply();
  //     if (i % 2 == 0) {
  //       domain_->insertNextLevelSetAsMaterial(ls, Material::SiO2);
  //     } else {
  //       domain_->insertNextLevelSetAsMaterial(ls, Material::Si3N4);
  //     }
  //   }

  //   if (holeRadius_ > 0. && maskHeight_ == 0.) {
  //     // cut out middle
  //     auto cutOut = lsDomainType::New(bounds_, boundaryConds_, gridDelta);
  //     origin_[D - 1] = 0.;
  //     viennals::MakeGeometry<NumericType, D>(
  //         cutOut,
  //         SmartPointer<viennals::Cylinder<NumericType, D>>::New(
  //             origin_, normal_, (numLayers_ + 1) * layerHeight_,
  //             holeRadius_))
  //         .apply();

  //     domain_->applyBooleanOperation(
  //         cutOut, viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT);

  //   } else if (trenchWidth_ > 0. && maskHeight_ == 0.) {
  //     auto cutOut = lsDomainType::New(bounds_, boundaryConds_, gridDelta);
  //     NumericType minPoint[D] = {
  //         static_cast<NumericType>(-trenchWidth_ / 2.),
  //         static_cast<NumericType>(-yExtent_ / 2. - gridDelta),
  //         (NumericType)0.};
  //     NumericType maxPoint[D] = {
  //         static_cast<NumericType>(trenchWidth_ / 2.),
  //         static_cast<NumericType>(yExtent_ / 2. + gridDelta),
  //         substrateHeight_ + layerHeight_ * numLayers_ + maskHeight_ +
  //             gridDelta};
  //     viennals::MakeGeometry<NumericType, D>(
  //         cutOut,
  //         SmartPointer<viennals::Box<NumericType, D>>::New(minPoint,
  //         maxPoint)) .apply();

  //     domain_->applyBooleanOperation(
  //         cutOut, viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT);
  //   }
  // }
};

} // namespace viennaps
