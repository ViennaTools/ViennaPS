#include <lsToSurfaceMesh.hpp>
#include <lsVTKWriter.hpp>
#include <psGDSReader.hpp>

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

  const NumericType gridDelta = 0.1;

  auto gds = psSmartPointer<psGDSGeometry<NumericType, D>>::New(gridDelta);
  std::string file = "polygons.gds";
  gds->setBoundaryPadding(1, 1);

  psGDSReader<NumericType, D>(gds, file).apply();
  for (int i = 0; i < 6; i++) {
    auto layer = gds->layerToLevelSet(i, 10., 0.5, true);
    printLS(layer, "poly_" + std::to_string(i) + ".vtp");
  }
  return 0;
}