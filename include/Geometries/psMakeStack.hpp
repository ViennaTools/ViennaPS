#pragma once

#include <lsBooleanOperation.hpp>
#include <lsDomain.hpp>
#include <lsMakeGeometry.hpp>
#include <lsToSurfaceMesh.hpp>
#include <lsToVoxelMesh.hpp>
#include <lsVTKWriter.hpp>
#include <lsWriteVisualizationMesh.hpp>

#include <psDomain.hpp>

/**
 * Creates a stack of alternating material layers with an etched
 * hole(3D)/trench(2D) in the middle.
 */
template <class T, int D> class psMakeStack {
  using PSPtrType = psSmartPointer<psDomain<T, D>>;
  using LSPtrType = psSmartPointer<lsDomain<T, D>>;

  const T gridDelta = 0.1;
  const T xExtent = 10;
  const T yExtent = 10;
  T bounds[2 * D];
  T normal[D];
  T origin[D] = {0.};

  const int numLayers = 11;
  const T layerHeight = 2;
  const T substrateHeight = 4;
  const T holeRadius = 0.;
  const T maskHeight = 0.;
  const bool periodicBoundary = false;

  typename lsDomain<T, D>::BoundaryType boundaryConds[D];

  PSPtrType geometry;

public:
  psMakeStack(PSPtrType passedDomain, const T passedGridDelta,
              const T passedXExtent, const T passedYExtent,
              const int passedNumLayers, const T passedLayerHeight,
              const T passedSubstrateHeight, const T passedHoleRadius,
              const T passedMaskHeight, const bool periodic = false)
      : geometry(passedDomain), gridDelta(passedGridDelta),
        xExtent(passedXExtent), yExtent(passedYExtent),
        numLayers(passedNumLayers), layerHeight(passedLayerHeight),
        substrateHeight(passedSubstrateHeight), holeRadius(passedHoleRadius),
        maskHeight(passedMaskHeight), periodicBoundary(periodic) {
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

  T getHeight() const { return substrateHeight + numLayers * layerHeight; }

private:
  void create2DGeometry() {
    geometry->clear();

    if (maskHeight > 0.) {
      // mask on top
      auto mask = LSPtrType::New(bounds, boundaryConds, gridDelta);
      origin[D - 1] = substrateHeight + layerHeight * numLayers + maskHeight;
      lsMakeGeometry<T, D>(mask,
                           lsSmartPointer<lsPlane<T, D>>::New(origin, normal))
          .apply();

      auto maskAdd = LSPtrType::New(bounds, boundaryConds, gridDelta);
      origin[D - 1] = substrateHeight + layerHeight * numLayers;
      normal[D - 1] = -1;
      lsMakeGeometry<T, D>(maskAdd,
                           lsSmartPointer<lsPlane<T, D>>::New(origin, normal))
          .apply();
      normal[D - 1] = 1.;

      lsBooleanOperation<T, D>(mask, maskAdd, lsBooleanOperationEnum::INTERSECT)
          .apply();

      T minPoint[D] = {-holeRadius,
                       substrateHeight + layerHeight * numLayers - gridDelta};
      T maxPoint[D] = {holeRadius, substrateHeight + layerHeight * numLayers +
                                       maskHeight + gridDelta};
      lsMakeGeometry<T, D>(maskAdd,
                           lsSmartPointer<lsBox<T, D>>::New(minPoint, maxPoint))
          .apply();

      lsBooleanOperation<T, D>(mask, maskAdd,
                               lsBooleanOperationEnum::RELATIVE_COMPLEMENT)
          .apply();

      geometry->insertNextLevelSetAsMaterial(mask, psMaterial::Mask);
    }

    // Silicon substrate
    auto substrate = LSPtrType::New(bounds, boundaryConds, gridDelta);
    origin[D - 1] = substrateHeight;
    lsMakeGeometry<T, D>(substrate,
                         lsSmartPointer<lsPlane<T, D>>::New(origin, normal))
        .apply();
    geometry->insertNextLevelSetAsMaterial(substrate, psMaterial::Si);

    // Si3N4/SiO2 layers
    T current = substrateHeight + layerHeight;
    for (int i = 0; i < numLayers; ++i) {
      auto ls = LSPtrType::New(bounds, boundaryConds, gridDelta);
      origin[D - 1] = substrateHeight + layerHeight * (i + 1);
      lsMakeGeometry<T, D>(ls,
                           lsSmartPointer<lsPlane<T, D>>::New(origin, normal))
          .apply();
      if (i % 2 == 0) {
        geometry->insertNextLevelSetAsMaterial(ls, psMaterial::SiO2);
      } else {
        geometry->insertNextLevelSetAsMaterial(ls, psMaterial::Si3N4);
      }
    }

    if (holeRadius > 0. && maskHeight == 0.) {
      // cut out middle
      auto cutOut = LSPtrType::New(bounds, boundaryConds, gridDelta);
      T minPoint[D] = {-holeRadius, 0.};
      T maxPoint[D] = {holeRadius, substrateHeight + layerHeight * numLayers +
                                       maskHeight + gridDelta};
      lsMakeGeometry<T, D>(cutOut,
                           lsSmartPointer<lsBox<T, D>>::New(minPoint, maxPoint))
          .apply();

      for (auto layer : *geometry->getLevelSets()) {
        lsBooleanOperation<T, D>(layer, cutOut,
                                 lsBooleanOperationEnum::RELATIVE_COMPLEMENT)
            .apply();
      }
    }
  }

  void create3DGeometry() {
    geometry->clear();

    if (maskHeight > 0.) {
      auto mask = LSPtrType::New(bounds, boundaryConds, gridDelta);
      origin[D - 1] = substrateHeight + layerHeight * numLayers + maskHeight;
      lsMakeGeometry<T, D>(mask,
                           lsSmartPointer<lsPlane<T, D>>::New(origin, normal))
          .apply();

      auto maskAdd = LSPtrType::New(bounds, boundaryConds, gridDelta);
      origin[D - 1] = substrateHeight + layerHeight * numLayers;
      normal[D - 1] = -1;
      lsMakeGeometry<T, D>(maskAdd,
                           lsSmartPointer<lsPlane<T, D>>::New(origin, normal))
          .apply();

      lsBooleanOperation<T, D>(mask, maskAdd, lsBooleanOperationEnum::INTERSECT)
          .apply();

      normal[D - 1] = 1.;
      lsMakeGeometry<T, D>(
          maskAdd, lsSmartPointer<lsCylinder<T, D>>::New(
                       origin, normal, maskHeight + gridDelta, holeRadius))
          .apply();

      lsBooleanOperation<T, D>(mask, maskAdd,
                               lsBooleanOperationEnum::RELATIVE_COMPLEMENT)
          .apply();

      geometry->insertNextLevelSetAsMaterial(mask, psMaterial::Mask);
    }

    // Silicon substrate
    auto substrate = LSPtrType::New(bounds, boundaryConds, gridDelta);
    origin[D - 1] = substrateHeight;
    lsMakeGeometry<T, D>(substrate,
                         lsSmartPointer<lsPlane<T, D>>::New(origin, normal))
        .apply();
    geometry->insertNextLevelSet(substrate, psMaterial::Si);

    // Si3N4/SiO2 layers
    for (int i = 0; i < numLayers; ++i) {
      auto ls = LSPtrType::New(bounds, boundaryConds, gridDelta);
      origin[D - 1] = substrateHeight + layerHeight * (i + 1);
      lsMakeGeometry<T, D>(ls,
                           lsSmartPointer<lsPlane<T, D>>::New(origin, normal))
          .apply();
      if (i % 2 == 0) {
        geometry->insertNextLevelSetAsMaterial(ls, psMaterial::SiO2);
      } else {
        geometry->insertNextLevelSetAsMaterial(ls, psMaterial::Si3N4);
      }
    }

    if (holeRadius > 0. && maskHeight == 0.) {
      // cut out middle
      auto cutOut = LSPtrType::New(bounds, boundaryConds, gridDelta);
      origin[D - 1] = 0.;
      lsMakeGeometry<T, D>(
          cutOut,
          lsSmartPointer<lsCylinder<T, D>>::New(
              origin, normal, (numLayers + 1) * layerHeight, holeRadius))
          .apply();

      for (auto layer : *geometry->getLevelSets()) {
        lsBooleanOperation<T, D>(layer, cutOut,
                                 lsBooleanOperationEnum::RELATIVE_COMPLEMENT)
            .apply();
      }
    }
  }

  void init() {
    bounds[0] = -xExtent;
    bounds[1] = xExtent;
    normal[0] = 0.;
    if (periodicBoundary)
      boundaryConds[0] = lsDomain<T, D>::BoundaryType::PERIODIC_BOUNDARY;
    else
      boundaryConds[0] = lsDomain<T, D>::BoundaryType::REFLECTIVE_BOUNDARY;

    if constexpr (D == 2) {
      normal[1] = 1.;
      bounds[2] = 0;
      bounds[3] = layerHeight * numLayers + gridDelta;
      boundaryConds[1] = lsDomain<T, D>::BoundaryType::INFINITE_BOUNDARY;
    } else {
      normal[1] = 0.;
      normal[2] = 1.;
      bounds[2] = -yExtent;
      bounds[3] = yExtent;
      bounds[4] = 0;
      bounds[5] = layerHeight * numLayers + gridDelta;
      if (periodicBoundary)
        boundaryConds[0] = lsDomain<T, D>::BoundaryType::PERIODIC_BOUNDARY;
      else
        boundaryConds[0] = lsDomain<T, D>::BoundaryType::REFLECTIVE_BOUNDARY;
      boundaryConds[2] = lsDomain<T, D>::BoundaryType::INFINITE_BOUNDARY;
    }
  }
};