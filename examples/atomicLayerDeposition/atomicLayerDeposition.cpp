#include <csMeanFreePath.hpp>
#include <csSegmentCells.hpp>
#include <psAtomicLayerModel.hpp>
#include <psDomain.hpp>
#include <psMakeTrench.hpp>
#include <psMeanFreePath.hpp>

#include "geometry.hpp"

int main(int argc, char *argv[]) {
  constexpr int D = 2;
  using NumericType = double;
  omp_set_num_threads(14);

  psLogger::setLogLevel(psLogLevel::INTERMEDIATE);

  // Parse the parameters
  psUtils::Parameters params;
  if (argc > 1) {
    params.readConfigFile(argv[1]);
  } else {
    std::cout << "Usage: " << argv[0] << " <config file>" << std::endl;
    return 1;
  }

  // Create a domain
  auto domain = psSmartPointer<psDomain<NumericType, D>>::New();
  if (params.get<int>("trench") > 0) {
    psMakeTrench<NumericType, D>(
        domain, params.get("gridDelta"),
        params.get("verticalWidth") + 2. * params.get("xPad"),
        params.get("verticalWidth") + 2. * params.get("xPad"),
        params.get("verticalWidth"), params.get("verticalDepth"), 0., 0., false,
        false, psMaterial::TiN)
        .apply();
  } else {
    makeLShape(domain, params, psMaterial::TiN);
  }
  domain->generateCellSet(params.get("verticalDepth") + params.get("topSpace"),
                          psMaterial::GAS, true);
  auto &cellSet = domain->getCellSet();
  csSegmentCells<NumericType, D>(cellSet).apply();

  psMeanFreePath<NumericType, D> mfpCalc;
  mfpCalc.setReflectionLimit(params.get<int>("reflectionLimit"));
  mfpCalc.setNumRaysPerPoint(params.get("raysPerPoint"));

  mfpCalc.setDomain(domain);
  mfpCalc.setMaterial(psMaterial::GAS);
  mfpCalc.setBulkLambda(params.get("bulkLambda"));
  mfpCalc.apply();

  cellSet->writeVTU("meanFreePath.vtu");

  // auto maxLambda = mfpCalc.getMaxLambda();
  // std::cout << "Max. mean free path: " << maxLambda << std::endl;

  psAtomicLayerModel<NumericType, D> model(domain);
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