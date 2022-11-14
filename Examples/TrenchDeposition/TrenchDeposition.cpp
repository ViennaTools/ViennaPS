#include <Geometries/psMakeTrench.hpp>
#include <SimpleDeposition.hpp>
#include <psConfigParser.hpp>
#include <psProcess.hpp>
#include <psToSurfaceMesh.hpp>
#include <psVTKWriter.hpp>
#include <psWriteVisualizationMesh.hpp>

int main(int argc, char *argv[]) {
  using NumericType = double;
  constexpr int D = 2;

  // Parse the parameters
  psProcessParameters<NumericType> params;
  if (argc > 1) {
    psConfigParser<NumericType> parser(argv[1]);
    parser.apply();
    params = parser.getParameters();
  }

  auto geometry = psSmartPointer<psDomain<NumericType, D>>::New();
  psMakeTrench<NumericType, D>(
      geometry, params.gridDelta /* grid delta */, params.xExtent /*x extent*/,
      params.yExtent /*y extent*/, params.trenchWidth /*trench width*/,
      params.trenchHeight /*trench height*/,
      params.taperAngle /* tapering angle */, false /*create mask*/)
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

  auto mesh = psSmartPointer<lsMesh<NumericType>>::New();
  psToSurfaceMesh<NumericType, D>(geometry, mesh).apply();
  psVTKWriter<NumericType>(mesh, "initial.vtp").apply();

  process.apply();

  psToSurfaceMesh<NumericType, D>(geometry, mesh).apply();
  psVTKWriter<NumericType>(mesh, "final.vtp").apply();

  if (D == 2)
    psWriteVisualizationMesh<NumericType, D>(geometry, "final").apply();

  return EXIT_SUCCESS;
}
