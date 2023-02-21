#pragma once

#include <lsBooleanOperation.hpp>
#include <lsDomain.hpp>
#include <lsMakeGeometry.hpp>
#include <lsToSurfaceMesh.hpp>
#include <lsToVoxelMesh.hpp>
#include <lsVTKWriter.hpp>
#include <lsWriteVisualizationMesh.hpp>

#include <psDomain.hpp>

template <class T, int D> class MakeStack {
  using PSPtrType = psSmartPointer<psDomain<T, D>>;
  using LSPtrType = psSmartPointer<lsDomain<T, D>>;

  T gridDelta = 0.1;
  T xExtent = 10;
  T yExtent = 10;
  T bounds[2 * D];
  T normal[D];
  T origin[D] = {0.};

  int numLayers = 11;
  T layerHeight = 2;
  T substrateHeight = 4;
  T holeRadius = 2;

  typename lsDomain<T, D>::BoundaryType boundaryConds[D];

public:
  MakeStack(const int passedLayer, const T passedGridDelta)
      : numLayers(passedLayer), gridDelta(passedGridDelta) {
    init();
  }

  PSPtrType makeStack() {
    if constexpr (D == 2) {
      return create2DGeometry();
    } else {
      return create3DGeometry();
    }
  }

  int getTopLayer() const { return numLayers + 1; }

  T getHeight() const { return substrateHeight + numLayers * layerHeight; }

  T getHoleRadius() const { return holeRadius; }

  void printGeometry(PSPtrType geometry, std::string fileName) {
    lsWriteVisualizationMesh<T, D> visMesh;
    for (auto ls : *geometry->getLevelSets()) {
      visMesh.insertNextLevelSet(ls);
    }
    visMesh.setFileName(fileName);
    visMesh.apply();
  }

  void printVoxelMesh(PSPtrType geometry, std::string fileName) {
    auto mesh = lsSmartPointer<lsMesh<T>>::New();
    lsToVoxelMesh<T, D> visMesh(mesh);
    for (auto ls : *geometry->getLevelSets()) {
      visMesh.insertNextLevelSet(ls);
    }
    visMesh.apply();
    lsVTKWriter<T>(mesh, fileName).apply();
  }

private:
  PSPtrType create2DGeometry() {

    // Silicon substrate
    auto substrate = LSPtrType::New(bounds, boundaryConds, gridDelta);
    origin[D - 1] = substrateHeight;
    auto plane = lsSmartPointer<lsPlane<T, D>>::New(origin, normal);
    lsMakeGeometry<T, D>(substrate, plane).apply();
    auto geometry = PSPtrType::New(substrate);

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

    // copy top layer for deposition
    auto depoLayer = LSPtrType::New(geometry->getLevelSets()->back());
    geometry->insertNextLevelSet(depoLayer);

    return geometry;
  }

  PSPtrType create3DGeometry() {

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
    auto geometry = PSPtrType::New(substrate);

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

    // copy top layer for deposition
    auto depoLayer = LSPtrType::New(geometry->getLevelSets()->back());
    geometry->insertNextLevelSet(depoLayer);

    return geometry;
  }

  void init() {
    bounds[0] = -xExtent;
    bounds[1] = xExtent;
    normal[0] = 0.;
    boundaryConds[0] = lsDomain<T, D>::BoundaryType::PERIODIC_BOUNDARY;

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
      boundaryConds[1] = lsDomain<T, D>::BoundaryType::PERIODIC_BOUNDARY;
      boundaryConds[2] = lsDomain<T, D>::BoundaryType::INFINITE_BOUNDARY;
    }
  }
};