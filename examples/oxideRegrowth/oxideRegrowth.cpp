#include <psMakeStack.hpp>
#include <psProcess.hpp>
#include <psWriteVisualizationMesh.hpp>

#include <psOxideRegrowth.hpp>

#include "parameters.hpp"

int main(int argc, char **argv) {
  using NumericType = double;

  omp_set_num_threads(12);
  constexpr int D = 2;

  psLogger::setLogLevel(psLogLevel::INTERMEDIATE);

  Parameters<NumericType> params;
  if (argc > 1) {
    auto config = psUtils::readConfigFile(argv[1]);
    if (config.empty()) {
      std::cerr << "Empty config provided" << std::endl;
      return -1;
    }
    params.fromMap(config);
  }

  const NumericType stability =
      2 * params.diffusionCoefficient /
      std::max(params.centerVelocity, params.scallopVelocity);
  std::cout << "Stability: " << stability << std::endl;
  constexpr NumericType timeStabilityFactor = D == 2 ? 0.245 : 0.145;
  if (0.5 * stability <= params.gridDelta) {
    std::cout << "Unstable parameters. Reduce grid spacing!" << std::endl;
    return -1;
  }

  auto domain = psSmartPointer<psDomain<NumericType, D>>::New();
  psMakeStack<NumericType, D>(domain, params.gridDelta, params.xExtent,
                              params.yExtent, params.numLayers,
                              params.layerHeight, params.substrateHeight,
                              0. /*hole radius*/, params.trenchWidth, 0., false)
      .apply();
  // copy top layer for deposition
  domain->duplicateTopLevelSet(psMaterial::Polymer);

  domain->generateCellSet(params.substrateHeight +
                              params.numLayers * params.layerHeight + 10.,
                          true /* true means cell set above surface */);
  auto &cellSet = domain->getCellSet();
  cellSet->addScalarData("byproductSum", 0.);
  cellSet->writeVTU("initial.vtu");
  if constexpr (D == 3) {
    std::array<bool, D> boundaryConds = {false};
    boundaryConds[1] = true;
    cellSet->setPeriodicBoundary(boundaryConds);
  }
  // we need neighborhood information for solving the
  // convection-diffusion equation on the cell set
  cellSet->buildNeighborhood();

  // The redeposition model captures byproducts from the selective etching
  // process in the cell set. The byproducts are then distributed by solving a
  // convection-diffusion equation on the cell set.
  auto model = psSmartPointer<psOxideRegrowth<NumericType, D>>::New(
      params.nitrideEtchRate / 60., params.oxideEtchRate / 60.,
      params.redepositionRate, params.redepositionThreshold,
      params.redepositionTimeInt, params.diffusionCoefficient, params.sink,
      params.scallopVelocity, params.centerVelocity,
      params.substrateHeight + params.numLayers * params.layerHeight,
      params.trenchWidth, timeStabilityFactor);

  psProcess<NumericType, D> process;
  process.setDomain(domain);
  process.setProcessModel(model);
  process.setProcessDuration(params.targetEtchDepth / params.nitrideEtchRate *
                             60.);
  process.setPrintTimeInterval(30);

  process.apply();

  psWriteVisualizationMesh<NumericType, D>(domain, "FinalStack").apply();
}