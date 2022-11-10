#include <psDomain.hpp>
#include <psGDSReader.hpp>
#include <psProcess.hpp>

template <class NumericType, int D>
void printLS(psSmartPointer<lsDomain<NumericType, D>> domain,
             std::string name) {
  auto mesh = lsSmartPointer<lsMesh<NumericType>>::New();
  lsToSurfaceMesh<NumericType, D>(domain, mesh).apply();
  lsVTKWriter<NumericType>(mesh, name).apply();
}

int main(int argc, char **argv) {
  using NumericType = double;
  constexpr int D = 3;

  // read GDS mask file
  const NumericType gridDelta = 0.01;
  auto mask = psSmartPointer<psGDSGeometry<NumericType, D>>::New(gridDelta);
  psGDSReader<NumericType, D>(mask, "SRAM_mask.gds").apply();

  // geometry setup
  auto maskBounds = mask->getBounds();
  NumericType bounds[2 * D] = {maskBounds[0][0],
                               maskBounds[1][0],
                               maskBounds[1][0],
                               maskBounds[1][1],
                               -10,
                               10};
  typename lsDomain<NumericType, D>::BoundaryType boundaryCons[D];
  for (int i = 0; i < D - 1; i++)
    boundaryCons[i] = lsDomain<NumericType, D>::BoundaryType::PERIODIC_BOUNDARY;
  boundaryCons[D - 1] =
      lsDomain<NumericType, D>::BoundaryType::INFINITE_BOUNDARY;
  auto geometry = psSmartPointer<psDomain<NumericType, D>>::New();

  // fin patterning
  {
    auto fins = mask->layerToLevelSet(0 /*layer*/, 0 /*base z position*/,
                                      0.1 /*height*/);
    geometry->insertNextLevelSet(fins);

    // substrate plane
    NumericType origin[D] = {0., 0., 0.};
    NumericType normal[D] = {0., 0., 1.};
    auto plane = psSmartPointer<lsDomain<NumericType, D>>::New(
        bounds, boundaryCons, gridDelta);
    lsMakeGeometry<NumericType, D>(
        plane, psSmartPointer<lsPlane<NumericType, D>>::New(origin, normal))
        .apply();

    geometry->insertNextLevelSet(plane);
  }

  {}

  return 0;
}