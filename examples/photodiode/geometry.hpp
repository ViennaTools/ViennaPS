#pragma once

#include <lsBooleanOperation.hpp>
#include <lsMakeGeometry.hpp>

#include <psDomain.hpp>

namespace ps = viennaps;

// Arbitrary materials, different from silicon and air
inline const ps::Material lensMaterial = ps::Material::Polymer;
inline const ps::Material passivationMaterial = ps::Material::SiN;
inline const ps::Material antiReflectMaterial1 = ps::Material::SiO2;
inline const ps::Material antiReflectMaterial2 = ps::Material::Si3N4;

template <class T, int D> class Builder {
  using lsDomainType = viennals::SmartPointer<viennals::Domain<T, D>>;
  using Boundary = typename viennals::Domain<T, D>::BoundaryType;

  T bounds[2 * D];
  Boundary bcs[D];
  T grid_dx;

public:
  Builder(T *domainBounds, Boundary *boundaryConds, T gridDelta)
      : grid_dx(gridDelta) {
    for (int i = 0; i < 2 * D; i++)
      bounds[i] = domainBounds[i];
    for (int i = 0; i < D; i++)
      bcs[i] = boundaryConds[i];
  }

  auto create() { return lsDomainType::New(bounds, bcs, grid_dx); }

  // Horizontal plane
  void makePlane(lsDomainType ls, T height, int orientation = 1) {
    T origin[D] = {};
    origin[D - 1] = height;
    T normal[D] = {};
    normal[D - 1] = orientation;

    viennals::MakeGeometry<T, D>(
        ls, viennals::SmartPointer<viennals::Plane<T, D>>::New(origin, normal))
        .apply();
  }

  auto createPlane(T height, int orientation = 1) {
    auto ls = create();
    makePlane(ls, height, orientation);
    return ls;
  }

  void makePlate(lsDomainType ls, T bottom, T height) {
    makePlane(ls, bottom + height);
    auto plateTop = createPlane(bottom, -1);

    intersect(ls, plateTop);
  }

  auto createPlate(T bottom, T height) {
    auto ls = create();
    makePlate(ls, bottom, height);
    return ls;
  }

  void makeBox(lsDomainType ls, T *minPoint, T *maxPoint) {
    viennals::MakeGeometry<T, D>(
        ls,
        viennals::SmartPointer<viennals::Box<T, D>>::New(minPoint, maxPoint))
        .apply();
  }

  void cutHole(lsDomainType plate, T holeWidth, T posX, T bottom, T height) {
    T minPoint[D] = {posX - holeWidth / 2, -holeWidth / 2};
    T maxPoint[D] = {posX + holeWidth / 2, holeWidth / 2};
    minPoint[D - 1] = bottom - grid_dx;
    maxPoint[D - 1] = bottom + height + grid_dx;

    auto cutout = create();
    makeBox(cutout, minPoint, maxPoint);

    exclude(plate, cutout);
  }

  // Evenly spaced holes through a plate
  void cutHoles(lsDomainType plate, int numHoles, T holeWidth, T minX, T maxX,
                T bottom, T height) {
    for (int i = 0; i < numHoles; i++) {
      T posX = minX + (maxX - minX) * (i + 1) / (numHoles + 1);
      cutHole(plate, holeWidth, posX, bottom, height);
    }
  }

  void combine(lsDomainType target, lsDomainType add) {
    viennals::BooleanOperation<T, D>(target, add,
                                     viennals::BooleanOperationEnum::UNION)
        .apply();
  }

  void intersect(lsDomainType target, lsDomainType bound) {
    viennals::BooleanOperation<T, D>(target, bound,
                                     viennals::BooleanOperationEnum::INTERSECT)
        .apply();
  }

  void exclude(lsDomainType target, lsDomainType subtract) {
    viennals::BooleanOperation<T, D>(
        target, subtract, viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT)
        .apply();
  }
};

template <class T, int D>
auto makeGeometry(T xExtent, T yExtent, T gridDelta, T bulkHeight, int numHoles,
                  T holeWidth, T holeDepth, T passivationHeight,
                  T antiReflectHeight1, T antiReflectHeight2, T maskHeight,
                  T baseHeight, bool withMask = false) {
  auto domain = ps::SmartPointer<ps::Domain<T, D>>::New();

  using Boundary = typename viennals::Domain<T, D>::BoundaryType;

  T bounds[2 * D] = {-xExtent / 2, xExtent / 2, -yExtent / 2, yExtent / 2};
  bounds[2 * D - 2] = baseHeight - gridDelta;
  bounds[2 * D - 1] = baseHeight + bulkHeight + passivationHeight +
                      (withMask ? maskHeight : T(0)) + gridDelta;

  Boundary bcs[D];
  for (int i = 0; i < D - 1; i++)
    bcs[i] = Boundary::REFLECTIVE_BOUNDARY;
  bcs[D - 1] = Boundary::INFINITE_BOUNDARY;

  Builder<T, D> builder(bounds, bcs, gridDelta);

  T bulkTop = baseHeight + bulkHeight;
  T roofBottom = withMask ? bulkTop : bulkTop - holeDepth;
  T roofHeight = withMask ? maskHeight : holeDepth;

  auto base = builder.createPlane(baseHeight);
  auto bulk = builder.createPlane(roofBottom);

  auto roof = builder.createPlate(roofBottom, roofHeight);
  builder.cutHoles(roof, numHoles, holeWidth, bounds[0], bounds[1], roofBottom,
                   roofHeight);

  auto passivation = builder.createPlane(bulkTop + passivationHeight);
  auto antiReflect2 =
      builder.createPlane(bulkTop + passivationHeight + antiReflectHeight2);
  auto antiReflect1 = builder.createPlane(
      bulkTop + passivationHeight + antiReflectHeight2 + antiReflectHeight1);

  domain->insertNextLevelSetAsMaterial(base, ps::Material::Si);
  if (!withMask)
    builder.combine(bulk, roof);
  domain->insertNextLevelSetAsMaterial(bulk, ps::Material::Si);
  if (withMask)
    domain->insertNextLevelSetAsMaterial(roof, ps::Material::Mask);

  domain->insertNextLevelSetAsMaterial(passivation, passivationMaterial, false);
  domain->insertNextLevelSetAsMaterial(antiReflect2, antiReflectMaterial2,
                                       true);
  domain->insertNextLevelSetAsMaterial(antiReflect1, antiReflectMaterial1,
                                       true);

  return domain;
}

// Returns a pair of arrays containing the top level-sets and their materials,
// removed from the domain
template <class Domain> auto extractTopLevelSets(Domain &domain, int num) {
  auto &levelSets = domain.getLevelSets();
  std::vector topLayers(levelSets.end() - num, levelSets.end());
  std::vector<ps::Material> topMaterials(num);

  for (int i = 0; i < num; ++i)
    topMaterials[num - 1 - i] =
        ps::Material{domain.getMaterialMap()->getMaterialMap()->getMaterialId(
            levelSets.size() - 1 - i)};
  for (int i = 0; i < num; ++i)
    domain.removeTopLevelSet();

  return std::pair{topLayers, topMaterials};
}
