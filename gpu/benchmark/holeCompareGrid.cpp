#include <geometries/psMakeHole.hpp>
#include <models/psIonBeamEtching.hpp>
#include <models/psMultiParticleProcess.hpp>
#include <models/psSF6O2Etching.hpp>
#include <models/psSingleParticleProcess.hpp>
#include <process/psProcess.hpp>

using namespace viennaps;

template <class NumericType, int D, class ModelType>
void runBenchmark(SmartPointer<Domain<NumericType, D>> geometry,
                  SmartPointer<ModelType> model,
                  const FluxEngineType fluxEngine,
                  const std::string &outputFilename,
                  const NumericType processDuration = 1.0) {
  auto copy = Domain<NumericType, D>::New(geometry);

  CoverageParameters coverageParams;
  coverageParams.tolerance = 1e-4;

  Process<NumericType, D> process(copy, model);
  process.setProcessDuration(processDuration);
  process.setFluxEngineType(fluxEngine);
  process.setParameters(coverageParams);

  Timer timer;
  timer.start();
  process.apply();
  timer.finish();

  std::cout << "Flux Engine: " << util::toString(fluxEngine)
            << ", Time taken: " << timer.currentDuration / 1e9 << " s"
            << std::endl;
  copy->saveSurfaceMesh(outputFilename);
}

int main(int argc, char *argv[]) {
  using NumericType = double;
  constexpr int D = 3;

  omp_set_num_threads(16);

  // set parameter units
  units::Length::setUnit(units::Length::MICROMETER);
  units::Time::setUnit(units::Time::MINUTE);
  Logger::setLogLevel(LogLevel::INFO);

  constexpr std::array<FluxEngineType, 2> fluxEngines = {
      FluxEngineType::GPU_TRIANGLE, FluxEngineType::GPU_DISK};
  constexpr std::array<NumericType, 4> gridDeltas = {0.04, 0.02, 0.01, 0.005};

  {
    auto modelParams = SF6O2Etching<NumericType, D>::defaultParameters();
    modelParams.etchantFlux = 4.5e3;
    modelParams.passivationFlux = 8e2;
    modelParams.ionFlux = 10.0;
    auto model = SmartPointer<SF6O2Etching<NumericType, D>>::New(modelParams);

    for (int i = 0; i < gridDeltas.size(); ++i) {
      auto gridDelta = gridDeltas[i];
      auto geometry = Domain<NumericType, D>::New(gridDelta, 1., 1.);
      MakeHole<NumericType, D>(geometry, 0.175, 0.0, 0.0, 1.2, 1.2,
                               HoleShape::QUARTER)
          .apply();

      for (const auto &fluxEngine : fluxEngines) {
        runBenchmark<NumericType, D>(geometry, model, fluxEngine,
                                     "result_PE_" + util::toString(fluxEngine) +
                                         "_" + std::to_string(i) + ".vtp",
                                     0.2);
      }
    }
  }

  if constexpr (false) {
    IBEParameters<NumericType> ibeParams;
    ibeParams.cos4Yield.isDefined = true;
    ibeParams.cos4Yield.a1 = 1.075;
    ibeParams.cos4Yield.a2 = -1.55;
    ibeParams.cos4Yield.a3 = 0.65;
    ibeParams.materialPlaneWaferRate[Material::Si] = 2.0;
    ibeParams.materialPlaneWaferRate[Material::Mask] = 0.15;
    ibeParams.thetaRMin = 60;
    auto model = SmartPointer<IonBeamEtching<NumericType, D>>::New(ibeParams);

    for (int i = 0; i < gridDeltas.size(); ++i) {
      auto gridDelta = gridDeltas[i];
      auto geometry = Domain<NumericType, D>::New(gridDelta, 1., 1.);
      MakeHole<NumericType, D>(geometry, 0.175, 0.0, 0.0, 1.2, 1.2,
                               HoleShape::FULL)
          .apply();

      for (const auto &fluxEngine : fluxEngines) {
        runBenchmark<NumericType, D>(geometry, model, fluxEngine,
                                     "result_IBE_" +
                                         util::toString(fluxEngine) + "_" +
                                         std::to_string(i) + ".vtp");
      }
    }
  }
}
