#include <Geometries/psMakeFin.hpp>
#include <PlasmaDamage.hpp>
#include <psConfigParser.hpp>
#include <psProcess.hpp>
#include <psToSurfaceMesh.hpp>
#include <psVTKWriter.hpp>

int main(int argc, char *argv[]) {
  using NumericType = float;
  constexpr int D = 3;

  // Parse the parameters
  psProcessParameters<NumericType> params;
  if (argc > 1) {
    psConfigParser<NumericType> parser(argv[1]);
    parser.apply();
    params = parser.getParameters();
  }

  auto geometry = psSmartPointer<psDomain<NumericType, D>>::New();
  psMakeFin<NumericType, D>(
      geometry, params.gridDelta /*grid delta*/, params.xExtent /*x extent*/,
      params.yExtent /*y extent*/, params.finWidth /*fin width*/,
      params.finHeight /*fin height*/, false /*create mask*/)
      .apply();
  // generate cell set with depth 5 below the lowest point of the surface
  geometry->generateCellSet(5. /*depth*/, false /*cell set below surface*/);

  PlasmaDamage<NumericType, D> model(
      params.ionEnergy /*mean ion energy (eV)*/,
      params.meanFreePath /* damage ion mean free path */,
      -1 /*mask material ID (no mask)*/);

  psProcess<NumericType, D> process;
  process.setDomain(geometry);
  process.setProcessModel(model.getProcessModel());
  process.setProcessDuration(0); // apply only damage model

  process.apply();

  geometry->getCellSet()->writeVTU("DamageModel.vtu");

  return EXIT_SUCCESS;
}
