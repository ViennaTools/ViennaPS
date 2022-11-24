#include <Geometries/psMakeFin.hpp>
#include <PlasmaDamage.hpp>
#include <psProcess.hpp>
#include <psToSurfaceMesh.hpp>
#include <psVTKWriter.hpp>

#include "Parameters.hpp"

int main(int argc, char *argv[]) {
  using NumericType = float;
  constexpr int D = 3;

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
  psMakeFin<NumericType, D>(geometry, params.gridDelta, params.xExtent,
                            params.yExtent, params.finWidth, params.finHeight,
                            false /*create mask*/)
      .apply();

  // generate cell set with depth 5 below the lowest point of the surface
  geometry->generateCellSet(5. /*depth*/, false /*cell set below surface*/);

  PlasmaDamage<NumericType, D> model(
      params.ionEnergy /* mean ion energy (eV) */,
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
