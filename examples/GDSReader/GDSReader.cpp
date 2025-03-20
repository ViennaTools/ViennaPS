#include <psDomain.hpp>
#include <psGDSReader.hpp>

namespace ps = viennaps;
namespace ls = viennals;

int main(int argc, char **argv) {
  using NumericType = double;
  constexpr int D = 3;

  // read GDS mask file
  constexpr NumericType gridDelta = 0.01;
  ls::BoundaryConditionEnum boundaryConds[D] = {
      ls::BoundaryConditionEnum::REFLECTIVE_BOUNDARY,
      ls::BoundaryConditionEnum::REFLECTIVE_BOUNDARY,
      ls::BoundaryConditionEnum::INFINITE_BOUNDARY};
  auto mask = ps::SmartPointer<ps::GDSGeometry<NumericType, D>>::New(gridDelta);
  mask->setBoundaryConditions(boundaryConds);
  ps::GDSReader<NumericType, D>(mask, "mask.gds").apply();

  // geometry setup
  auto bounds = mask->getBounds();
  auto geometry = ps::SmartPointer<ps::Domain<NumericType, D>>::New();

  // substrate plane
  NumericType origin[D] = {0., 0., 0.};
  NumericType normal[D] = {0., 0., 1.};
  auto plane = ps::SmartPointer<ls::Domain<NumericType, D>>::New(
      bounds, boundaryConds, gridDelta);
  ls::MakeGeometry<NumericType, D>(
      plane, ps::SmartPointer<ls::Plane<NumericType, D>>::New(origin, normal))
      .apply();

  geometry->insertNextLevelSet(plane);

  auto layer0 =
      mask->layerToLevelSet(0 /*layer*/, 0 /*base z position*/, 0.1 /*height*/);
  geometry->insertNextLevelSet(layer0);

//   auto layer1 = mask->layerToLevelSet(1 /*layer*/, -0.15 /*base z position*/,
//                                       0.45 /*height*/);
//   geometry->insertNextLevelSet(layer1);

  geometry->saveSurfaceMesh("Geometry.vtp", true /* add material IDs */);

  return 0;
}