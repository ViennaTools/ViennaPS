#pragma once

#include <csDenseCellSet.hpp>

#include <lsBooleanOperation.hpp>
#include <lsMakeGeometry.hpp>

#include <vcUtil.hpp>

namespace geometry {

using namespace viennals;
using namespace viennacs;

template <class T, int D> using levelSetType = SmartPointer<Domain<T, D>>;
template <class T, int D> using levelSetsType = std::vector<levelSetType<T, D>>;
using materialMapType = SmartPointer<MaterialMap>;

template <class T, int D>
void addLevelSet(levelSetsType<T, D> &levelSets, levelSetType<T, D> levelSet,
                 materialMapType matMap, std::vector<int> &materials, int material,
                 bool wrapLowerLevelSet = true) {
  if (!levelSets.empty() && wrapLowerLevelSet) {
    BooleanOperation<T, D>(levelSet, levelSets.back(),
                           BooleanOperationEnum::UNION)
        .apply();
  }

  levelSets.push_back(levelSet);
  if (matMap) matMap->insertNextMaterial(material);
  materials.push_back(material);
}

template <class T, int D>
void makePlane(levelSetType<T, D> &domain, const T *origin, const T *normal) {
  MakeGeometry<T, D>(domain, SmartPointer<Plane<T, D>>::New(origin, normal))
      .apply();
}

template <class T, int D>
void makeBox(levelSetType<T, D> &domain, const T *minPoint, const T *maxPoint) {
  MakeGeometry<T, D> maker(domain, SmartPointer<Box<T, D>>::New(minPoint, maxPoint));
  maker.setIgnoreBoundaryConditions(true);
  maker.apply();
}

template <class T, int D>
void makeCylinder(levelSetType<T, D> &domain, const T *origin, const T *axis,
                  T height, T radius) {
  MakeGeometry<T, D> maker(domain, SmartPointer<Cylinder<T, D>>::New(origin, axis, height, radius));
  maker.setIgnoreBoundaryConditions(true);
  maker.apply();
}

template <class T, int D>
auto makeStructure(util::Parameters &params, materialMapType matMap, std::vector<int> &materials,
                   int substrateMaterial, int maskMaterial, int ambientMaterial) {
  const T gridDelta = params.get("gridDelta");
  const T substrateHeight = params.get("substrateHeight");
  const T ambientHeight = params.get("ambientHeight");
  const T maskHeight = params.get("maskHeight");
  const T holeRadius = params.get("holeRadius");
  BoundaryConditionEnum boundaryConds[D] = {
      BoundaryConditionEnum::REFLECTIVE_BOUNDARY,
      BoundaryConditionEnum::REFLECTIVE_BOUNDARY};
  boundaryConds[D - 1] = BoundaryConditionEnum::INFINITE_BOUNDARY;
  T bounds[2 * D] = {-params.get("xExtent") / 2., params.get("xExtent") / 2.,
                     -params.get("yExtent") / 2., params.get("yExtent") / 2.};
  bounds[2 * D - 2] = 0.;
  bounds[2 * D - 1] = substrateHeight + maskHeight + ambientHeight + gridDelta;

  T origin[D] = {};
  T normal[D] = {};
  normal[D - 1] = 1.;

  levelSetsType<T, D> levelSets;

  // Substrate
  origin[D - 1] = 0.;
  // origin[D - 1] = substrateHeight;
  if constexpr (D == 2) {
    auto bottom = levelSetType<T, D>::New(bounds, boundaryConds, gridDelta);
    makePlane(bottom, origin, normal);
    addLevelSet(levelSets, bottom, matMap, materials, substrateMaterial);

    auto substrate = levelSetType<T, D>::New(bounds, boundaryConds, gridDelta);
    makePlane(substrate, origin + substrateHeight, normal);
    addLevelSet(levelSets, substrate, matMap, materials, substrateMaterial);
  } else {
    auto substrate = levelSetType<T, D>::New(bounds, boundaryConds, gridDelta);
    makeCylinder(substrate, origin, normal, substrateHeight, params.get("xExtent") / 2.0);
    addLevelSet(levelSets, substrate, matMap, materials, substrateMaterial);
  }

  // Mask
  if (maskHeight > 0.) {
    auto mask = levelSetType<T, D>::New(bounds, boundaryConds, gridDelta);

    if constexpr (D == 3) {
      origin[D - 1] = substrateHeight;
      makeCylinder(mask, origin, normal, maskHeight, params.get("xExtent") / 2.0);

      auto maskHole = levelSetType<T, D>::New(bounds, boundaryConds, gridDelta);
      makeCylinder(maskHole, origin, normal, maskHeight + 2 * gridDelta, holeRadius);

      BooleanOperation<T, D>(mask, maskHole,
                             BooleanOperationEnum::RELATIVE_COMPLEMENT)
          .apply();
    } else {
      origin[D - 1] = substrateHeight + maskHeight;
      makePlane(mask, origin, normal);

      auto maskAdd = levelSetType<T, D>::New(bounds, boundaryConds, gridDelta);

      T minPoint[D] = {-holeRadius, -holeRadius};
      T maxPoint[D] = {holeRadius, holeRadius};
      minPoint[D - 1] = substrateHeight - gridDelta;
      maxPoint[D - 1] = substrateHeight + maskHeight + gridDelta;

      makeBox(maskAdd, minPoint, maxPoint);

      BooleanOperation<T, D>(mask, maskAdd,
                             BooleanOperationEnum::RELATIVE_COMPLEMENT)
          .apply();
    }

    addLevelSet(levelSets, mask, matMap, materials, maskMaterial);
  }

  // Ambient
  if (ambientHeight > 0.) {
    auto ambient = levelSetType<T, D>::New(bounds, boundaryConds, gridDelta);
    if constexpr (D == 3) {
      origin[D - 1] = substrateHeight;
      makeCylinder(ambient, origin, normal, ambientHeight, params.get("xExtent") / 2.0);
    } else {
      origin[D - 1] = substrateHeight + maskHeight + ambientHeight;
      makePlane(ambient, origin, normal);
    }

    addLevelSet(levelSets, ambient, matMap, materials, ambientMaterial);
  }

  return levelSets;
}

template <class T, int D>
void makeCellSet(DenseCellSet<T, D> &cellSet, util::Parameters &params,
                 int substrateMaterial, int maskMaterial, int ambientMaterial) {
  // Generate Geometry (Level Sets)
  auto matMap = SmartPointer<MaterialMap>::New();
  std::vector<int> materials;
  auto levelSets = makeStructure<T, D>(params, matMap, materials, substrateMaterial,
                                       maskMaterial, ambientMaterial);

  // Create Cell Set (Discretization)
  T depth = params.get("substrateHeight") + params.get("ambientHeight");
  cellSet.setCellSetPosition(true);
  cellSet.fromLevelSets(levelSets, matMap, depth);
}

} // namespace geometry
