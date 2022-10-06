#include <lsToSurfaceMesh.hpp>
#include <lsVTKWriter.hpp>
#include <psGDSReader.hpp>

int main(int argc, char **argv) {
  using NumericType = double;
  constexpr int D = 3;

  auto gds = psSmartPointer<psGDSGeometry<NumericType, D>>::New(0.05);
  std::string file = "polygons.gds";
  gds->setBoundaryPadding(1, 1);

  psGDSReader<NumericType, D>(gds, file).apply();

  for (int i = 2; i < 3; i++) {
    auto layer = gds->layerToLevelSet(i, 10., 0.5, true);

    auto mesh = lsSmartPointer<lsMesh<NumericType>>::New();
    lsToSurfaceMesh<NumericType, D>(layer, mesh).apply();
    lsVTKWriter<NumericType>(mesh, "poly_" + std::to_string(i) + ".vtp")
        .apply();
  }
  return 0;
}