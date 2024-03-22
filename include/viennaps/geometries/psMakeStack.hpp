#pragma once

#include "../psDomain.hpp"

#include <lsBooleanOperation.hpp>
#include <lsDomain.hpp>
#include <lsMakeGeometry.hpp>

/// Generates a stack of alternating SiO2/Si3N4 layers featuring an optionally
/// etched hole (3D) or trench (2D) at the center. The stack emerges in the
/// positive z direction (3D) or y direction (2D) and is centered around the
/// origin, with its x/y extent specified. Users have the flexibility to
/// introduce periodic boundaries in the x and y directions. Additionally, the
/// stack can incorporate a top mask with a central hole of a specified radius
/// or a trench with a designated width. This versatile functionality enables
/// users to create diverse and customized structures for simulation scenarios.
template <class NumericType, int D> class psMakeStack {
  using PSPtrType = psSmartPointer<psDomain<NumericType, D>>;
  using LSPtrType = psSmartPointer<lsDomain<NumericType, D>>;

  PSPtrType domain = nullptr;

  const NumericType gridDelta;
  const NumericType xExtent;
  const NumericType yExtent;
  double bounds[2 * D];
  NumericType normal[D];
  NumericType origin[D] = {0.};

  const int numLayers;
  const NumericType layerHeight;
  const NumericType substrateHeight;
  NumericType holeRadius;
  const NumericType trenchWidth;
  const NumericType maskHeight;
  const bool periodicBoundary = false;

  typename lsDomain<NumericType, D>::BoundaryType boundaryConds[D];

public:
  psMakeStack(PSPtrType passedDomain, const NumericType passedGridDelta,
              const NumericType passedXExtent, const NumericType passedYExtent,
              const int passedNumLayers, const NumericType passedLayerHeight,
              const NumericType passedSubstrateHeight,
              const NumericType passedHoleRadius,
              const NumericType passedTrenchWidth,
              const NumericType passedMaskHeight, const bool periodic = false)
      : domain(passedDomain), gridDelta(passedGridDelta),
        xExtent(passedXExtent), yExtent(passedYExtent),
        numLayers(passedNumLayers), layerHeight(passedLayerHeight),
        substrateHeight(passedSubstrateHeight), holeRadius(passedHoleRadius),
        trenchWidth(passedTrenchWidth), maskHeight(passedMaskHeight),
        periodicBoundary(periodic) {
    init();
  }

  void apply() {
    if constexpr (D == 2) {
      create2DGeometry();
    } else {
      create3DGeometry();
    }
  }

  int getTopLayer() const { return numLayers; }

  NumericType getHeight() const {
    return substrateHeight + numLayers * layerHeight;
  }

private:
  void create2DGeometry() {
    domain->clear();

    if (maskHeight > 0.) {
      // mask on top
      auto mask = LSPtrType::New(bounds, boundaryConds, gridDelta);
      origin[D - 1] = substrateHeight + layerHeight * numLayers + maskHeight;
      lsMakeGeometry<NumericType, D>(
          mask, lsSmartPointer<lsPlane<NumericType, D>>::New(origin, normal))
          .apply();

      auto maskAdd = LSPtrType::New(bounds, boundaryConds, gridDelta);
      origin[D - 1] = substrateHeight + layerHeight * numLayers;
      normal[D - 1] = -1;
      lsMakeGeometry<NumericType, D>(
          maskAdd, lsSmartPointer<lsPlane<NumericType, D>>::New(origin, normal))
          .apply();
      normal[D - 1] = 1.;

      lsBooleanOperation<NumericType, D>(mask, maskAdd,
                                         lsBooleanOperationEnum::INTERSECT)
          .apply();

      if (holeRadius == 0.) {
        holeRadius = trenchWidth / 2.;
      }
      NumericType minPoint[D] = {
          -holeRadius, substrateHeight + layerHeight * numLayers - gridDelta};
      NumericType maxPoint[D] = {holeRadius, substrateHeight +
                                                 layerHeight * numLayers +
                                                 maskHeight + gridDelta};
      lsMakeGeometry<NumericType, D>(
          maskAdd,
          lsSmartPointer<lsBox<NumericType, D>>::New(minPoint, maxPoint))
          .apply();

      lsBooleanOperation<NumericType, D>(
          mask, maskAdd, lsBooleanOperationEnum::RELATIVE_COMPLEMENT)
          .apply();

      domain->insertNextLevelSetAsMaterial(mask, psMaterial::Mask);
    }

    // Silicon substrate
    auto substrate = LSPtrType::New(bounds, boundaryConds, gridDelta);
    origin[D - 1] = substrateHeight;
    lsMakeGeometry<NumericType, D>(
        substrate, lsSmartPointer<lsPlane<NumericType, D>>::New(origin, normal))
        .apply();
    domain->insertNextLevelSetAsMaterial(substrate, psMaterial::Si);

    // Si3N4/SiO2 layers
    NumericType current = substrateHeight + layerHeight;
    for (int i = 0; i < numLayers; ++i) {
      auto ls = LSPtrType::New(bounds, boundaryConds, gridDelta);
      origin[D - 1] = substrateHeight + layerHeight * (i + 1);
      lsMakeGeometry<NumericType, D>(
          ls, lsSmartPointer<lsPlane<NumericType, D>>::New(origin, normal))
          .apply();
      if (i % 2 == 0) {
        domain->insertNextLevelSetAsMaterial(ls, psMaterial::SiO2);
      } else {
        domain->insertNextLevelSetAsMaterial(ls, psMaterial::Si3N4);
      }
    }

    if ((holeRadius > 0. || trenchWidth > 0.) && maskHeight == 0.) {
      if (holeRadius == 0.) {
        holeRadius = trenchWidth / 2.;
      }
      // cut out middle
      auto cutOut = LSPtrType::New(bounds, boundaryConds, gridDelta);
      NumericType minPoint[D] = {-holeRadius, 0.};
      NumericType maxPoint[D] = {holeRadius, substrateHeight +
                                                 layerHeight * numLayers +
                                                 maskHeight + gridDelta};
      lsMakeGeometry<NumericType, D>(
          cutOut,
          lsSmartPointer<lsBox<NumericType, D>>::New(minPoint, maxPoint))
          .apply();

      for (auto layer : *domain->getLevelSets()) {
        lsBooleanOperation<NumericType, D>(
            layer, cutOut, lsBooleanOperationEnum::RELATIVE_COMPLEMENT)
            .apply();
      }
    }
  }

  void create3DGeometry() {
    domain->clear();

    if (maskHeight > 0.) {
      auto mask = LSPtrType::New(bounds, boundaryConds, gridDelta);
      origin[D - 1] = substrateHeight + layerHeight * numLayers + maskHeight;
      lsMakeGeometry<NumericType, D>(
          mask, lsSmartPointer<lsPlane<NumericType, D>>::New(origin, normal))
          .apply();

      auto maskAdd = LSPtrType::New(bounds, boundaryConds, gridDelta);
      origin[D - 1] = substrateHeight + layerHeight * numLayers;
      normal[D - 1] = -1;
      lsMakeGeometry<NumericType, D>(
          maskAdd, lsSmartPointer<lsPlane<NumericType, D>>::New(origin, normal))
          .apply();

      lsBooleanOperation<NumericType, D>(mask, maskAdd,
                                         lsBooleanOperationEnum::INTERSECT)
          .apply();

      if (holeRadius > 0.) {
        normal[D - 1] = 1.;
        lsMakeGeometry<NumericType, D>(
            maskAdd, lsSmartPointer<lsCylinder<NumericType, D>>::New(
                         origin, normal, maskHeight + gridDelta, holeRadius))
            .apply();

        lsBooleanOperation<NumericType, D>(
            mask, maskAdd, lsBooleanOperationEnum::RELATIVE_COMPLEMENT)
            .apply();
      } else if (trenchWidth > 0.) {
        NumericType minPoint[D] = {
            static_cast<NumericType>(-trenchWidth / 2.),
            static_cast<NumericType>(-yExtent / 2. - gridDelta), origin[D - 1]};
        NumericType maxPoint[D] = {
            static_cast<NumericType>(trenchWidth / 2.),
            static_cast<NumericType>(yExtent / 2. + gridDelta),
            origin[D - 1] + maskHeight + gridDelta};
        lsMakeGeometry<NumericType, D>(
            maskAdd,
            lsSmartPointer<lsBox<NumericType, D>>::New(minPoint, maxPoint))
            .apply();

        lsBooleanOperation<NumericType, D>(
            mask, maskAdd, lsBooleanOperationEnum::RELATIVE_COMPLEMENT)
            .apply();
      }

      domain->insertNextLevelSetAsMaterial(mask, psMaterial::Mask);
    }

    // Silicon substrate
    auto substrate = LSPtrType::New(bounds, boundaryConds, gridDelta);
    origin[D - 1] = substrateHeight;
    lsMakeGeometry<NumericType, D>(
        substrate, lsSmartPointer<lsPlane<NumericType, D>>::New(origin, normal))
        .apply();
    domain->insertNextLevelSetAsMaterial(substrate, psMaterial::Si);

    // Si3N4/SiO2 layers
    for (int i = 0; i < numLayers; ++i) {
      auto ls = LSPtrType::New(bounds, boundaryConds, gridDelta);
      origin[D - 1] = substrateHeight + layerHeight * (i + 1);
      lsMakeGeometry<NumericType, D>(
          ls, lsSmartPointer<lsPlane<NumericType, D>>::New(origin, normal))
          .apply();
      if (i % 2 == 0) {
        domain->insertNextLevelSetAsMaterial(ls, psMaterial::SiO2);
      } else {
        domain->insertNextLevelSetAsMaterial(ls, psMaterial::Si3N4);
      }
    }

    if (holeRadius > 0. && maskHeight == 0.) {
      // cut out middle
      auto cutOut = LSPtrType::New(bounds, boundaryConds, gridDelta);
      origin[D - 1] = 0.;
      lsMakeGeometry<NumericType, D>(
          cutOut,
          lsSmartPointer<lsCylinder<NumericType, D>>::New(
              origin, normal, (numLayers + 1) * layerHeight, holeRadius))
          .apply();

      for (auto layer : *domain->getLevelSets()) {
        lsBooleanOperation<NumericType, D>(
            layer, cutOut, lsBooleanOperationEnum::RELATIVE_COMPLEMENT)
            .apply();
      }
    } else if (trenchWidth > 0. && maskHeight == 0.) {
      auto cutOut = LSPtrType::New(bounds, boundaryConds, gridDelta);
      NumericType minPoint[D] = {
          static_cast<NumericType>(-trenchWidth / 2.),
          static_cast<NumericType>(-yExtent / 2. - gridDelta), (NumericType)0.};
      NumericType maxPoint[D] = {
          static_cast<NumericType>(trenchWidth / 2.),
          static_cast<NumericType>(yExtent / 2. + gridDelta),
          substrateHeight + layerHeight * numLayers + maskHeight + gridDelta};
      lsMakeGeometry<NumericType, D>(
          cutOut,
          lsSmartPointer<lsBox<NumericType, D>>::New(minPoint, maxPoint))
          .apply();

      for (auto layer : *domain->getLevelSets()) {
        lsBooleanOperation<NumericType, D>(
            layer, cutOut, lsBooleanOperationEnum::RELATIVE_COMPLEMENT)
            .apply();
      }
    }
  }

  void init() {
    bounds[0] = -xExtent / 2.;
    bounds[1] = xExtent / 2.;
    normal[0] = 0.;
    if (periodicBoundary)
      boundaryConds[0] =
          lsDomain<NumericType, D>::BoundaryType::PERIODIC_BOUNDARY;
    else
      boundaryConds[0] =
          lsDomain<NumericType, D>::BoundaryType::REFLECTIVE_BOUNDARY;

    if constexpr (D == 2) {
      normal[1] = 1.;
      bounds[2] = 0;
      bounds[3] = layerHeight * numLayers + gridDelta;
      boundaryConds[1] =
          lsDomain<NumericType, D>::BoundaryType::INFINITE_BOUNDARY;
    } else {
      normal[1] = 0.;
      normal[2] = 1.;
      bounds[2] = -yExtent / 2.;
      bounds[3] = yExtent / 2.;
      bounds[4] = 0;
      bounds[5] = layerHeight * numLayers + gridDelta;
      if (periodicBoundary)
        boundaryConds[1] =
            lsDomain<NumericType, D>::BoundaryType::PERIODIC_BOUNDARY;
      else
        boundaryConds[1] =
            lsDomain<NumericType, D>::BoundaryType::REFLECTIVE_BOUNDARY;
      boundaryConds[2] =
          lsDomain<NumericType, D>::BoundaryType::INFINITE_BOUNDARY;
    }
  }
};
