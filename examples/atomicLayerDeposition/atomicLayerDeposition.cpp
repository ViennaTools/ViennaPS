#include <cellSet/csAtomicLayerProcess.hpp>
#include <cellSet/csMeanFreePath.hpp>
#include <cellSet/csSegmentCells.hpp>
#include <geometries/psMakeHole.hpp>

#include <psDomain.hpp>
#include <psMeanFreePath.hpp>
#include <psUtils.hpp>

#include "geometry.hpp"

namespace viennaps = ps;

int main(int argc, char *argv[]) {
  constexpr int D = 2;
  using NumericType = double;

  ps::Logger::setLogLevel(ps::LogLevel::INTERMEDIATE);

  // Parse the parameters
  ps::utils::Parameters params;
  if (argc > 1) {
    params.readConfigFile(argv[1]);
  } else {
    std::cout << "Usage: " << argv[0] << " <config file>" << std::endl;
    return 1;
  }
  omp_set_num_threads(params.get<int>("numThreads"));

  // Create a domain
  auto domain = ps::SmartPointer<ps::Domain<NumericType, D>>::New();
  if (params.get<int>("trench") > 0) {
    ps::MakeHole<NumericType, D>(
        domain, params.get("gridDelta"),
        2 * params.get("verticalWidth") + 2. * params.get("xPad"),
        2 * params.get("verticalWidth") + 2. * params.get("xPad"),
        params.get("verticalWidth"), params.get("verticalDepth"), 0., 0., false,
        false, ps::Material::TiN)
        .apply();
  } else {
    makeLShape(domain, params, ps::Material::TiN);
  }
  // Generate the cell set from the domain
  domain->generateCellSet(params.get("verticalDepth") + params.get("topSpace"),
                          ps::Material::GAS, true);
  auto &cellSet = domain->getCellSet();
  // Segment the cells into surface, material, and gas cells
  csSegmentCells<NumericType, D>(cellSet).apply();

  cellSet->writeVTU("initial.vtu");

  psUtils::Timer timer;
  timer.start();

  // Calculate the mean free path for the gas cells
  psMeanFreePath<NumericType, D> mfpCalc;
  mfpCalc.setDomain(domain);
  mfpCalc.setNumRaysPerCell(params.get("raysPerCell"));
  mfpCalc.setReflectionLimit(params.get<int>("reflectionLimit"));
  mfpCalc.setRngSeed(params.get<int>("seed"));
  mfpCalc.setMaterial(psMaterial::GAS);
  mfpCalc.setBulkLambda(params.get("bulkLambda"));
  mfpCalc.apply();

  timer.finish();
  std::cout << "Mean free path calculation took " << timer.totalDuration * 1e-9
            << " seconds." << std::endl;

  csAtomicLayerProcess<NumericType, D> model(domain);
  model.setMaxLambda(params.get("bulkLambda"));
  model.setPrintInterval(params.get("printInterval"));
  model.setStabilityFactor(params.get("stabilityFactor"));
  model.setFirstPrecursor("H2O", params.get("H2O_meanThermalVelocity"),
                          params.get("H2O_adsorptionRate"),
                          params.get("H2O_desorptionRate"),
                          params.get("p1_time"), params.get("inFlux"));
  model.setSecondPrecursor("TMA", params.get("TMA_meanThermalVelocity"),
                           params.get("TMA_adsorptionRate"),
                           params.get("TMA_desorptionRate"),
                           params.get("p2_time"), params.get("inFlux"));
  model.setPurgeParameters(params.get("purge_meanThermalVelocity"),
                           params.get("purge_time"));
  // The deposition probability is (H2O_cov * TMA_cov)^order
  model.setReactionOrder(params.get("reactionOrder"));

  model.apply();
}
