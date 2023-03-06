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
  const T holeRadius = 2;
  const bool periodicBoundary = false;

  typename lsDomain<T, D>::BoundaryType boundaryConds[D];

  PSPtrType geometry;

public:
  psMakeStack(PSPtrType passedDomain, const int passedNumLayers,
              const T passedGridDelta)
      : numLayers(passedNumLayers), gridDelta(passedGridDelta),
        geometry(passedDomain) {
    init();
  }

  psMakeStack(PSPtrType passedDomain, const T passedGridDelta,
              const T passedXExtent, const T passedYExtent,
              const int passedNumLayers, const T passedLayerHeight,
              const T passedSubstrateHeight, const T passedHoleRadius,
              const bool periodic = false)
      : geometry(passedDomain), gridDelta(passedGridDelta),
        xExtent(passedXExtent), yExtent(passedYExtent),
        numLayers(passedNumLayers), layerHeight(passedLayerHeight),
        substrateHeight(passedSubstrateHeight), holeRadius(passedHoleRadius),
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

  int getTopLayer() const { return numLayers + 1; }

  T getHeight() const { return substrateHeight + numLayers * layerHeight; }

  T getHoleRadius() const { return holeRadius; }

private:
  void create2DGeometry() {
    geometry->clear();

    // Silicon substrate
    auto substrate = LSPtrType::New(bounds, boundaryConds, gridDelta);
    origin[D - 1] = substrateHeight;
    auto plane = lsSmartPointer<lsPlane<T, D>>::New(origin, normal);
    lsMakeGeometry<T, D>(substrate, plane).apply();

    // Si3N4/SiO2 layers
    T current = substrateHeight + layerHeight;
    for (int i = 0; i < numLayers; ++i) {
      auto ls = LSPtrType::New(bounds, boundaryConds, gridDelta);
      origin[D - 1] = substrateHeight + layerHeight * (i + 1);
      auto plane = lsSmartPointer<lsPlane<T, D>>::New(origin, normal);
      lsMakeGeometry<T, D>(ls, plane).apply();
      geometry->insertNextLevelSet(ls);
    }

    // cut out middle
    auto cutOut = LSPtrType::New(bounds, boundaryConds, gridDelta);
    T minPoint[D] = {-holeRadius, substrateHeight * 0.8};
    T maxPoint[D] = {holeRadius,
                     substrateHeight + layerHeight * numLayers + gridDelta};
    lsMakeGeometry<T, D>(cutOut,
                         lsSmartPointer<lsBox<T, D>>::New(minPoint, maxPoint))
        .apply();

    for (auto layer : *geometry->getLevelSets()) {
      lsBooleanOperation<T, D>(layer, cutOut,
                               lsBooleanOperationEnum::RELATIVE_COMPLEMENT)
          .apply();
    }
  }

  void create3DGeometry() {
    geometry->clear();

    // cut out middle
    auto cutOut = LSPtrType::New(bounds, boundaryConds, gridDelta);
    origin[D - 1] = substrateHeight * 0.8;
    lsMakeGeometry<T, D>(
        cutOut, lsSmartPointer<lsCylinder<T, D>>::New(
                    origin, normal, (numLayers + 1) * layerHeight, holeRadius))
        .apply();

    // Silicon substrate
    auto substrate = LSPtrType::New(bounds, boundaryConds, gridDelta);
    origin[D - 1] = substrateHeight;
    auto plane = lsSmartPointer<lsPlane<T, D>>::New(origin, normal);
    lsMakeGeometry<T, D>(substrate, plane).apply();
    lsBooleanOperation<T, D>(substrate, cutOut,
                             lsBooleanOperationEnum::RELATIVE_COMPLEMENT)
        .apply();

    // Si3N4/SiO2 layers
    for (int i = 0; i < numLayers; ++i) {
      auto ls = LSPtrType::New(bounds, boundaryConds, gridDelta);
      origin[D - 1] = substrateHeight + layerHeight * (i + 1);
      auto plane = lsSmartPointer<lsPlane<T, D>>::New(origin, normal);
      lsMakeGeometry<T, D>(ls, plane).apply();
      lsBooleanOperation<T, D>(ls, cutOut,
                               lsBooleanOperationEnum::RELATIVE_COMPLEMENT)
          .apply();
      geometry->insertNextLevelSet(ls);
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