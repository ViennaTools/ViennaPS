#include <geometries/psMakeTrench.hpp>
#include <lsCompareChamfer.hpp>
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

  NumericType gridDelta = 1.5;
  NumericType xExtent = 50.0;
  NumericType yExtent = 20.0;
  NumericType trenchWidth = 30.0;
  NumericType maskHeight = 10.0;

  domain->setup(gridDelta, xExtent, yExtent, BoundaryType::REFLECTIVE_BOUNDARY);
  MakeTrench<NumericType, D>(domain, trenchWidth, 0.0, 0.0, maskHeight, 5.0,
                             true)
      .apply();
  domain->saveSurfaceMesh("initial.vtp", true);

  auto domainFE = Domain<NumericType, D>::New(domain);
  auto domainFEAdapt = Domain<NumericType, D>::New(domain);
  auto domainRK2 = Domain<NumericType, D>::New(domain);
  auto domainRK3 = Domain<NumericType, D>::New(domain);

  // Parameters
  NumericType depThickness = 2.1;
  NumericType isoEtchDepth = 1.0;
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
  auto spatialScheme = viennals::SpatialSchemeEnum::LAX_FRIEDRICHS_1ST_ORDER;
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
  process.setProcessDuration(0.3);
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

  CoverageParameters coverage;
  coverage.tolerance = 1e-4;
  process.setParameters(coverage);

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

  auto runSimulation = [&](SmartPointer<Domain<NumericType, D>> currDomain,
                           viennals::TemporalSchemeEnum scheme, bool adaptive,
                           unsigned subdivisions, std::string name) {
    advection.temporalScheme = scheme;
    advection.adaptiveTimeStepping = adaptive;
    advection.adaptiveTimeStepSubdivisions = subdivisions;
    advection.calculateIntermediateVelocities =
        intermediateVelocityCalculations;
    process.setParameters(advection);

    process.setDomain(currDomain);
    depProcess.setDomain(currDomain);
    chemEtchProcess.setDomain(currDomain);

    currDomain->duplicateTopLevelSet(Material::Polymer);

    viennacore::Timer timer;
    timer.start();
    for (int j = 0; j < numCycles; ++j) {
      depProcess.apply();
      process.apply();
      chemEtchProcess.apply();
    }
    timer.finish();
    std::cout << name << " Time: " << timer.currentDuration / 1e6 << " ms"
              << std::endl;

    currDomain->saveSurfaceMesh("final_result_" + name + ".vtp", true);
  };

  // 1. FE without adaptive time stepping
  runSimulation(domainFE, viennals::TemporalSchemeEnum::FORWARD_EULER, false, 1,
                "FE");

  // 2. FE with adaptive time stepping (true, 100)
  runSimulation(domainFEAdapt, viennals::TemporalSchemeEnum::FORWARD_EULER,
                true, 50, "FE_Adapt");

  // 3. RK2 without adaptive time stepping
  runSimulation(domainRK2, viennals::TemporalSchemeEnum::RUNGE_KUTTA_2ND_ORDER,
                false, 1, "RK2");

  // 4. RK3 without adaptive time stepping
  runSimulation(domainRK3, viennals::TemporalSchemeEnum::RUNGE_KUTTA_3RD_ORDER,
                false, 1, "RK3");

  // Compare Chamfer distances
  viennals::CompareChamfer<NumericType, D> chamferFE_Adapt(
      domainFE->getLevelSets().back(), domainFEAdapt->getLevelSets().back());
  chamferFE_Adapt.apply();
  // std::cout << "Chamfer FE vs FE_Adapt: "
  //           << chamferFE_Adapt.getChamferDistance() << std::endl;

  viennals::CompareChamfer<NumericType, D> chamferFE_RK2(
      domainFE->getLevelSets().back(), domainRK2->getLevelSets().back());
  chamferFE_RK2.apply();
  // std::cout << "Chamfer FE vs RK2: " << chamferFE_RK2.getChamferDistance()
  //           << std::endl;

  viennals::CompareChamfer<NumericType, D> chamferFE_RK3(
      domainFE->getLevelSets().back(), domainRK3->getLevelSets().back());
  chamferFE_RK3.apply();
  // std::cout << "Chamfer FE vs RK3: " << chamferFE_RK3.getChamferDistance()
  //           << std::endl;
}

} // namespace viennacore

int main() { viennacore::RunTest<double, 2>(); }
