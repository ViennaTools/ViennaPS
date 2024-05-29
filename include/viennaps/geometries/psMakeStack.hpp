#pragma once

#include "../psDomain.hpp"

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
  using lsDomainType = SmartPointer<viennals::Domain<NumericType, D>>;
  using BoundaryEnum = typename viennals::Domain<NumericType, D>::BoundaryType;

  psDomainType pDomain_ = nullptr;

  const NumericType gridDelta_;
  const NumericType xExtent_;
  const NumericType yExtent_;
  double bounds_[2 * D];
  NumericType normal_[D];
  NumericType origin_[D] = {0.};

  const int numLayers_;
  const NumericType layerHeight_;
  const NumericType substrateHeight_;
  NumericType holeRadius_;
  const NumericType trenchWidth_;
  const NumericType maskHeight_;
  const bool periodicBoundary_ = false;

  BoundaryEnum boundaryConds_[D];

public:
  MakeStack(psDomainType domain, NumericType gridDelta, NumericType xExtent,
            NumericType yExtent, int numLayers, NumericType layerHeight,
            NumericType substrateHeight, NumericType holeRadius,
            NumericType trenchWidth, NumericType maskHeight,
            bool periodicBoundary = false)
      : pDomain_(domain), gridDelta_(gridDelta), xExtent_(xExtent),
        yExtent_(yExtent), numLayers_(numLayers), layerHeight_(layerHeight),
        substrateHeight_(substrateHeight), holeRadius_(holeRadius),
        trenchWidth_(trenchWidth), maskHeight_(maskHeight),
        periodicBoundary_(periodicBoundary) {
    init();
  }

  void apply() {
    if constexpr (D == 2) {
      create2DGeometry();
    } else {
      create3DGeometry();
    }
  }

  int getTopLayer() const { return numLayers_; }

  NumericType getHeight() const {
    return substrateHeight_ + numLayers_ * layerHeight_;
  }

private:
  void create2DGeometry() {
    pDomain_->clear();

    if (maskHeight_ > 0.) {
      // mask on top
      auto mask = lsDomainType::New(bounds_, boundaryConds_, gridDelta_);
      origin_[D - 1] =
          substrateHeight_ + layerHeight_ * numLayers_ + maskHeight_;
      viennals::MakeGeometry<NumericType, D>(
          mask,
          SmartPointer<viennals::Plane<NumericType, D>>::New(origin_, normal_))
          .apply();

      auto maskAdd = lsDomainType::New(bounds_, boundaryConds_, gridDelta_);
      origin_[D - 1] = substrateHeight_ + layerHeight_ * numLayers_;
      normal_[D - 1] = -1;
      viennals::MakeGeometry<NumericType, D>(
          maskAdd,
          SmartPointer<viennals::Plane<NumericType, D>>::New(origin_, normal_))
          .apply();
      normal_[D - 1] = 1.;

      viennals::BooleanOperation<NumericType, D>(
          mask, maskAdd, viennals::BooleanOperationEnum::INTERSECT)
          .apply();

      if (holeRadius_ == 0.) {
        holeRadius_ = trenchWidth_ / 2.;
      }
      NumericType minPoint[D] = {-holeRadius_, substrateHeight_ +
                                                   layerHeight_ * numLayers_ -
                                                   gridDelta_};
      NumericType maxPoint[D] = {holeRadius_, substrateHeight_ +
                                                  layerHeight_ * numLayers_ +
                                                  maskHeight_ + gridDelta_};
      viennals::MakeGeometry<NumericType, D>(
          maskAdd,
          SmartPointer<viennals::Box<NumericType, D>>::New(minPoint, maxPoint))
          .apply();

      viennals::BooleanOperation<NumericType, D>(
          mask, maskAdd, viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT)
          .apply();

      pDomain_->insertNextLevelSetAsMaterial(mask, Material::Mask);
    }

    // Silicon substrate
    auto substrate = lsDomainType::New(bounds_, boundaryConds_, gridDelta_);
    origin_[D - 1] = substrateHeight_;
    viennals::MakeGeometry<NumericType, D>(
        substrate,
        SmartPointer<viennals::Plane<NumericType, D>>::New(origin_, normal_))
        .apply();
    pDomain_->insertNextLevelSetAsMaterial(substrate, Material::Si);

    // Si3N4/SiO2 layers
    NumericType current = substrateHeight_ + layerHeight_;
    for (int i = 0; i < numLayers_; ++i) {
      auto ls = lsDomainType::New(bounds_, boundaryConds_, gridDelta_);
      origin_[D - 1] = substrateHeight_ + layerHeight_ * (i + 1);
      viennals::MakeGeometry<NumericType, D>(
          ls,
          SmartPointer<viennals::Plane<NumericType, D>>::New(origin_, normal_))
          .apply();
      if (i % 2 == 0) {
        pDomain_->insertNextLevelSetAsMaterial(ls, Material::SiO2);
      } else {
        pDomain_->insertNextLevelSetAsMaterial(ls, Material::Si3N4);
      }
    }

    if ((holeRadius_ > 0. || trenchWidth_ > 0.) && maskHeight_ == 0.) {
      if (holeRadius_ == 0.) {
        holeRadius_ = trenchWidth_ / 2.;
      }
      // cut out middle
      auto cutOut = lsDomainType::New(bounds_, boundaryConds_, gridDelta_);
      NumericType minPoint[D] = {-holeRadius_, 0.};
      NumericType maxPoint[D] = {holeRadius_, substrateHeight_ +
                                                  layerHeight_ * numLayers_ +
                                                  maskHeight_ + gridDelta_};
      viennals::MakeGeometry<NumericType, D>(
          cutOut,
          SmartPointer<viennals::Box<NumericType, D>>::New(minPoint, maxPoint))
          .apply();

      for (auto layer : *pDomain_->getLevelSets()) {
        viennals::BooleanOperation<NumericType, D>(
            layer, cutOut, viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT)
            .apply();
      }
    }
  }

  void create3DGeometry() {
    pDomain_->clear();

    if (maskHeight_ > 0.) {
      auto mask = lsDomainType::New(bounds_, boundaryConds_, gridDelta_);
      origin_[D - 1] =
          substrateHeight_ + layerHeight_ * numLayers_ + maskHeight_;
      viennals::MakeGeometry<NumericType, D>(
          mask,
          SmartPointer<viennals::Plane<NumericType, D>>::New(origin_, normal_))
          .apply();

      auto maskAdd = lsDomainType::New(bounds_, boundaryConds_, gridDelta_);
      origin_[D - 1] = substrateHeight_ + layerHeight_ * numLayers_;
      normal_[D - 1] = -1;
      viennals::MakeGeometry<NumericType, D>(
          maskAdd,
          SmartPointer<viennals::Plane<NumericType, D>>::New(origin_, normal_))
          .apply();

      viennals::BooleanOperation<NumericType, D>(
          mask, maskAdd, viennals::BooleanOperationEnum::INTERSECT)
          .apply();

      if (holeRadius_ > 0.) {
        normal_[D - 1] = 1.;
        viennals::MakeGeometry<NumericType, D>(
            maskAdd,
            SmartPointer<viennals::Cylinder<NumericType, D>>::New(
                origin_, normal_, maskHeight_ + gridDelta_, holeRadius_))
            .apply();

        viennals::BooleanOperation<NumericType, D>(
            mask, maskAdd, viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT)
            .apply();
      } else if (trenchWidth_ > 0.) {
        NumericType minPoint[D] = {
            static_cast<NumericType>(-trenchWidth_ / 2.),
            static_cast<NumericType>(-yExtent_ / 2. - gridDelta_),
            origin_[D - 1]};
        NumericType maxPoint[D] = {
            static_cast<NumericType>(trenchWidth_ / 2.),
            static_cast<NumericType>(yExtent_ / 2. + gridDelta_),
            origin_[D - 1] + maskHeight_ + gridDelta_};
        viennals::MakeGeometry<NumericType, D>(
            maskAdd, SmartPointer<viennals::Box<NumericType, D>>::New(minPoint,
                                                                      maxPoint))
            .apply();

        viennals::BooleanOperation<NumericType, D>(
            mask, maskAdd, viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT)
            .apply();
      }

      pDomain_->insertNextLevelSetAsMaterial(mask, Material::Mask);
    }

    // Silicon substrate
    auto substrate = lsDomainType::New(bounds_, boundaryConds_, gridDelta_);
    origin_[D - 1] = substrateHeight_;
    viennals::MakeGeometry<NumericType, D>(
        substrate,
        SmartPointer<viennals::Plane<NumericType, D>>::New(origin_, normal_))
        .apply();
    pDomain_->insertNextLevelSetAsMaterial(substrate, Material::Si);

    // Si3N4/SiO2 layers
    for (int i = 0; i < numLayers_; ++i) {
      auto ls = lsDomainType::New(bounds_, boundaryConds_, gridDelta_);
      origin_[D - 1] = substrateHeight_ + layerHeight_ * (i + 1);
      viennals::MakeGeometry<NumericType, D>(
          ls,
          SmartPointer<viennals::Plane<NumericType, D>>::New(origin_, normal_))
          .apply();
      if (i % 2 == 0) {
        pDomain_->insertNextLevelSetAsMaterial(ls, Material::SiO2);
      } else {
        pDomain_->insertNextLevelSetAsMaterial(ls, Material::Si3N4);
      }
    }

    if (holeRadius_ > 0. && maskHeight_ == 0.) {
      // cut out middle
      auto cutOut = lsDomainType::New(bounds_, boundaryConds_, gridDelta_);
      origin_[D - 1] = 0.;
      viennals::MakeGeometry<NumericType, D>(
          cutOut,
          SmartPointer<viennals::Cylinder<NumericType, D>>::New(
              origin_, normal_, (numLayers_ + 1) * layerHeight_, holeRadius_))
          .apply();

      for (auto layer : *pDomain_->getLevelSets()) {
        viennals::BooleanOperation<NumericType, D>(
            layer, cutOut, viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT)
            .apply();
      }
    } else if (trenchWidth_ > 0. && maskHeight_ == 0.) {
      auto cutOut = lsDomainType::New(bounds_, boundaryConds_, gridDelta_);
      NumericType minPoint[D] = {
          static_cast<NumericType>(-trenchWidth_ / 2.),
          static_cast<NumericType>(-yExtent_ / 2. - gridDelta_),
          (NumericType)0.};
      NumericType maxPoint[D] = {
          static_cast<NumericType>(trenchWidth_ / 2.),
          static_cast<NumericType>(yExtent_ / 2. + gridDelta_),
          substrateHeight_ + layerHeight_ * numLayers_ + maskHeight_ +
              gridDelta_};
      viennals::MakeGeometry<NumericType, D>(
          cutOut,
          SmartPointer<viennals::Box<NumericType, D>>::New(minPoint, maxPoint))
          .apply();

      for (auto layer : *pDomain_->getLevelSets()) {
        viennals::BooleanOperation<NumericType, D>(
            layer, cutOut, viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT)
            .apply();
      }
    }
  }

  void init() {
    bounds_[0] = -xExtent_ / 2.;
    bounds_[1] = xExtent_ / 2.;
    normal_[0] = 0.;
    if (periodicBoundary_)
      boundaryConds_[0] = BoundaryEnum::PERIODIC_BOUNDARY;
    else
      boundaryConds_[0] = BoundaryEnum::REFLECTIVE_BOUNDARY;

    if constexpr (D == 2) {
      normal_[1] = 1.;
      bounds_[2] = 0;
      bounds_[3] = layerHeight_ * numLayers_ + gridDelta_;
      boundaryConds_[1] = BoundaryEnum::INFINITE_BOUNDARY;
    } else {
      normal_[1] = 0.;
      normal_[2] = 1.;
      bounds_[2] = -yExtent_ / 2.;
      bounds_[3] = yExtent_ / 2.;
      bounds_[4] = 0;
      bounds_[5] = layerHeight_ * numLayers_ + gridDelta_;
      if (periodicBoundary_)
        boundaryConds_[1] = BoundaryEnum::PERIODIC_BOUNDARY;
      else
        boundaryConds_[1] = BoundaryEnum::REFLECTIVE_BOUNDARY;
      boundaryConds_[2] = BoundaryEnum::INFINITE_BOUNDARY;
    }
  }
};

} // namespace viennaps
