#include <lsToSurfaceMesh.hpp>
#include <lsVTKWriter.hpp>
#include <psGDSReader.hpp>

int main(int argc, char **argv) {
  using NumericType = double;
  constexpr int D = 3;

  auto gds = psSmartPointer<psGDSGeometry<NumericType, D>>::New();
  std::string file = "polygons.gds";

  psGDSReader<NumericType, D>(gds, file).apply();

  gds->setBoundaryPadding(0.5, 0.5);

  for (int i = 0; i < 5; i++) {
    auto layer = gds->layerToLevelSet(i, 0.5, 0.05, true);

    auto mesh = lsSmartPointer<lsMesh<NumericType>>::New();
    lsToSurfaceMesh<NumericType, D>(layer, mesh).apply();
    lsVTKWriter<NumericType>(mesh, "gds_test_new_" + std::to_string(i) + ".vtp")
        .apply();
  }
  return 0;
}