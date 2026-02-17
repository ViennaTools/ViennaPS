#include <geometries/psMakeTrench.hpp>
#include <lsTestAsserts.hpp>
#include <vcTestAsserts.hpp>
#include <viennaps.hpp>

namespace viennacore {

using namespace viennaps;

template <class NumericType, int D> void RunTest() {
  // Set log level
  Logger::setLogLevel(LogLevel::WARNING);

  // Set units
  units::Length::setUnit("nm");
  units::Time::setUnit("s");

  // Load initial domain
  auto domain = Domain<NumericType, D>::New();

  NumericType gridDelta = 1.0;
  NumericType xExtent = 100.0;
  NumericType yExtent = 25.0;
  NumericType trenchWidth = 50.0;
  NumericType maskHeight = 15.0;

  domain->setup(gridDelta, xExtent, yExtent, BoundaryType::REFLECTIVE_BOUNDARY);
  MakeTrench<NumericType, D>(domain, trenchWidth, 0.0, 0.0, maskHeight, 5.0,
                             true)
      .apply();
  domain->saveSurfaceMesh("initial.vtp", true);

  // Parameters
  NumericType depThickness = 2.1;
  NumericType isoEtchDepth = 2.0;
  double ionFlux = 85.71261101803212;
  double etchantFlux = 5401.405552767991;
  double meanEnergy = 387.0;
  double ionExponent = 200.0;
  double siAie = 5.915203825312938;
  double kSigma = 1000.0;

  // Options
  int rpp = 200;
  int smoothingNeighbors = 2;
  double timeStepRatio = 0.49;
  bool adaptiveTimeStepping = false;
  bool intermediateVelocityCalculations = true;
  auto spatialScheme = viennals::SpatialSchemeEnum::ENGQUIST_OSHER_1ST_ORDER;
  auto temporalScheme = viennals::TemporalSchemeEnum::RUNGE_KUTTA_2ND_ORDER;
  auto fluxEngineType = FluxEngineType::CPU_TRIANGLE;

  // Setup SF6C4F8Etching Model
  auto modelParams = SF6C4F8Etching<NumericType, D>::defaultParameters();
  modelParams.ionFlux = ionFlux;
  modelParams.etchantFlux = etchantFlux;
  modelParams.Ions.meanEnergy = meanEnergy;
  modelParams.Ions.exponent = ionExponent;
  modelParams.Substrate.A_ie = siAie;
  modelParams.passivationFlux = 0.0;
  modelParams.Substrate.k_sigma = kSigma;

  auto model = SmartPointer<SF6C4F8Etching<NumericType, D>>::New(modelParams);
  model->setProcessName("PT-Step");

  // Setup Plasma Etch Process
  Process<NumericType, D> process;
  process.setDomain(domain);
  process.setProcessModel(model);
  process.setProcessDuration(0.33);
  process.setFluxEngineType(fluxEngineType);

  RayTracingParameters rayTracing;
  rayTracing.raysPerPoint = rpp;
  rayTracing.smoothingNeighbors = smoothingNeighbors;
  process.setParameters(rayTracing);

  AdvectionParameters advection;
  advection.timeStepRatio = timeStepRatio;
  advection.adaptiveTimeStepping = adaptiveTimeStepping;
  advection.spatialScheme = spatialScheme;
  advection.temporalScheme = temporalScheme;
  advection.calculateIntermediateVelocities = intermediateVelocityCalculations;
  process.setParameters(advection);

  // Setup Isotropic Deposition
  auto isoDepModel = SmartPointer<IsotropicProcess<NumericType, D>>::New(1.0);
  isoDepModel->setProcessName("IsotropicDeposition");

  Process<NumericType, D> depProcess;
  depProcess.setDomain(domain);
  depProcess.setProcessModel(isoDepModel);
  depProcess.setProcessDuration(depThickness);

  // Setup Chemical Etch (Directional Process with Isotropic Velocity)
  std::array<NumericType, 3> directionDown = {0.0, -1.0, 0.0};
  viennaps::impl::RateSet<NumericType> etchDirDown(
      directionDown, 0.0, -isoEtchDepth, {Material::Mask, Material::Polymer},
      false);

  auto chemEtchModel =
      SmartPointer<DirectionalProcess<NumericType, D>>::New(etchDirDown);

  Process<NumericType, D> chemEtchProcess;
  chemEtchProcess.setDomain(domain);
  chemEtchProcess.setProcessModel(chemEtchModel);
  chemEtchProcess.setProcessDuration(1.0);

  // Run Sequence
  int numCycles = 1;
  domain->duplicateTopLevelSet(Material::Polymer);

  for (int j = 0; j < numCycles; ++j) {
    std::cout << "Cycle " << j + 1 << "/" << numCycles << std::endl;
    // Deposition
    std::cout << "  Deposition..." << std::endl;
    depProcess.apply();

    // Plasma Etch
    std::cout << "  Plasma Etch..." << std::endl;
    process.apply();

    // Chemical Etch
    std::cout << "  Chemical Etch..." << std::endl;
    chemEtchProcess.apply();

    domain->saveSurfaceMesh("result_at_cycle_" + std::to_string(j) + ".vtp",
                            true);
  }

  domain->removeTopLevelSet();

  Planarize<NumericType, D>(domain, 0.0).apply();

  domain->saveSurfaceMesh("final_result.vtp", true);

  VC_TEST_ASSERT(domain->getLevelSets().size() >= 2);
  LSTEST_ASSERT_VALID_LS(domain->getLevelSets().back(), NumericType, D);
}

} // namespace viennacore

int main() { VC_RUN_2D_TESTS }
