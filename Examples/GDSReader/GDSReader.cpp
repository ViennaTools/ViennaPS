#include <psDomain.hpp>
#include <psGDSReader.hpp>
#include <psPointValuesToLevelSet.hpp>
#include <psProcess.hpp>
#include <psToDiskMesh.hpp>
#include <psVTKWriter.hpp>

int main(int argc, char **argv) {
  using NumericType = double;
  constexpr int D = 3;

  // read GDS mask file
  const NumericType gridDelta = 0.01;
  typename lsDomain<NumericType, D>::BoundaryType boundaryCons[D];
  for (int i = 0; i < D - 1; i++)
    boundaryCons[i] =
        lsDomain<NumericType, D>::BoundaryType::REFLECTIVE_BOUNDARY;
  boundaryCons[D - 1] =
      lsDomain<NumericType, D>::BoundaryType::INFINITE_BOUNDARY;
  auto mask = psSmartPointer<psGDSGeometry<NumericType, D>>::New(gridDelta);
  mask->setBoundaryConditions(boundaryCons);
  psGDSReader<NumericType, D>(mask, "mask.gds").apply();

  // geometry setup
  double *bounds = mask->getBounds();
  auto geometry = psSmartPointer<psDomain<NumericType, D>>::New();

  // substrate plane
  NumericType origin[D] = {0., 0., 0.};
  NumericType normal[D] = {0., 0., 1.};
  auto plane = psSmartPointer<lsDomain<NumericType, D>>::New(
      bounds, boundaryCons, gridDelta);
  lsMakeGeometry<NumericType, D>(
      plane, psSmartPointer<lsPlane<NumericType, D>>::New(origin, normal))
      .apply();

  geometry->insertNextLevelSet(plane);

  auto layer0 =
      mask->layerToLevelSet(0 /*layer*/, 0 /*base z position*/, 0.1 /*height*/);
  geometry->insertNextLevelSet(layer0);

  auto layer1 = mask->layerToLevelSet(1 /*layer*/, -0.15 /*base z position*/,
                                      0.45 /*height*/);
  geometry->insertNextLevelSet(layer1);

  geometry->printSurface("Geometry.vtp", true /* add material IDs */);

  return 0;
}