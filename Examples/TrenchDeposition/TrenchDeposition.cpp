#include <Geometries/psMakeTrench.hpp>
#include <SimpleDeposition.hpp>
#include <psProcess.hpp>
#include <psToSurfaceMesh.hpp>
#include <psUtils.hpp>
#include <psVTKWriter.hpp>
#include <psWriteVisualizationMesh.hpp>

#include "Parameters.hpp"

int main(int argc, char *argv[]) {
  using NumericType = double;
  constexpr int D = 2;

  // Parse the parameters
  Parameters<NumericType> params;
  if (argc > 1) {
    auto config = psUtils::readConfigFile(argv[1]);
    if (config.empty()) {
      std::cerr << "Empty config provided" << std::endl;
      return -1;
    }
    params.fromMap(config);
  }

  auto geometry = psSmartPointer<psDomain<NumericType, D>>::New();
  psMakeTrench<NumericType, D>(
      geometry, params.gridDelta /* grid delta */, params.xExtent /*x extent*/,
      params.yExtent /*y extent*/, params.trenchWidth /*trench width*/,
      params.trenchHeight /*trench height*/,
      params.taperAngle /* tapering angle */)
      .apply();

  // copy top layer to capture deposition
  auto depoLayer = psSmartPointer<lsDomain<NumericType, D>>::New(
      geometry->getLevelSets()->back());
  geometry->insertNextLevelSet(depoLayer);

  SimpleDeposition<NumericType, D> model(
      params.stickingProbability /*particle sticking probability*/,
      params.sourcePower /*particel source power*/);

  psProcess<NumericType, D> process;
  process.setDomain(geometry);
  process.setProcessModel(model.getProcessModel());
  process.setNumberOfRaysPerPoint(1000);
  process.setProcessDuration(params.processTime);

  geometry->printSurface("initial.vtp");

  process.apply();

  geometry->printSurface("final.vtp");

  if constexpr (D == 2)
    psWriteVisualizationMesh<NumericType, D>(geometry, "final").apply();

  return EXIT_SUCCESS;
}
