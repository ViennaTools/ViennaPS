#include <geometries/psMakeStack.hpp>
#include <psProcess.hpp>

#include <models/psOxideRegrowth.hpp>

namespace ps = viennaps;

int main(int argc, char **argv) {
  using NumericType = double;

  omp_set_num_threads(12);
  constexpr int D = 2;

  ps::Logger::setLogLevel(ps::LogLevel::INTERMEDIATE);

  // Parse the parameters
  ps::util::Parameters params;
  if (argc > 1) {
    params.readConfigFile(argv[1]);
  } else {
    std::cout << "Usage: " << argv[0] << " <config file>" << std::endl;
    return 1;
  }

  const NumericType stability =
      2 * params.get("diffusionCoefficient") /
      std::max(params.get("centerVelocity"), params.get("scallopVelocity"));
  std::cout << "Stability: " << stability << std::endl;
  constexpr NumericType timeStabilityFactor = D == 2 ? 0.245 : 0.145;
  if (0.5 * stability <= params.get("gridDelta")) {
    std::cout << "Unstable parameters. Reduce grid spacing!" << std::endl;
    return -1;
  }

  auto domain = ps::SmartPointer<ps::Domain<NumericType, D>>::New(
      params.get("gridDelta"), params.get("xExtent"), params.get("yExtent"));
  ps::MakeStack<NumericType, D>(
      domain, params.get<int>("numLayers"), params.get("layerHeight"),
      params.get("substrateHeight"), 0. /*holeRadius*/,
      params.get("trenchWidth"), 0. /*maskHeight*/, 0. /*taperAngle*/)
      .apply();
  // copy top layer for deposition
  domain->duplicateTopLevelSet(ps::Material::Polymer);

  domain->generateCellSet(
      params.get("substrateHeight") +
          params.get("numLayers") * params.get("layerHeight") + 10.,
      ps::Material::GAS, true /* true means cell set above surface */);
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
  auto model = ps::SmartPointer<ps::OxideRegrowth<NumericType, D>>::New(
      params.get("nitrideEtchRate") / 60., params.get("oxideEtchRate") / 60.,
      params.get("redepositionRate"), params.get("redepositionThreshold"),
      params.get("redepositionTimeInt"), params.get("diffusionCoefficient"),
      params.get("sink"), params.get("scallopVelocity"),
      params.get("centerVelocity"),
      params.get("substrateHeight") +
          params.get("numLayers") * params.get("layerHeight"),
      params.get("trenchWidth"), timeStabilityFactor);

  ps::Process<NumericType, D> process;
  process.setDomain(domain);
  process.setProcessModel(model);
  process.setProcessDuration(params.get("targetEtchDepth") /
                             params.get("nitrideEtchRate") * 60.);
  process.apply();

  domain->saveVolumeMesh("finalStack");
}
