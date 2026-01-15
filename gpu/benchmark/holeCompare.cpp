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
                  const std::string &outputFilename) {
  auto copy = Domain<NumericType, D>::New(geometry);

  Process<NumericType, D> process(copy, model);
  process.setProcessDuration(1.0);
  process.setFluxEngineType(fluxEngine);

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

  // geometry setup
  auto geometry = Domain<NumericType, D>::New(0.03, 1., 1.);
  MakeHole<NumericType, D>(geometry, 0.175,
                           0.0, // holeDepth
                           0.0, // holeTaperAngle
                           1.2, 1.2, HoleShape::HALF)
      .apply();

  constexpr std::array<FluxEngineType, 4> fluxEngines = {
      FluxEngineType::GPU_TRIANGLE, FluxEngineType::GPU_DISK,
      FluxEngineType::CPU_TRIANGLE, FluxEngineType::CPU_DISK};

  {
    auto modelParams = SF6O2Etching<NumericType, D>::defaultParameters();
    modelParams.etchantFlux = 4.5e3;
    modelParams.passivationFlux = 8e2;
    modelParams.ionFlux = 10.0;
    auto model = SmartPointer<SF6O2Etching<NumericType, D>>::New(modelParams);

    for (const auto &fluxEngine : fluxEngines) {
      runBenchmark<NumericType, D>(geometry, model, fluxEngine,
                                   "result_PE_" + util::toString(fluxEngine) +
                                       ".vtp");
    }
  }

  {
    auto model = SmartPointer<SingleParticleProcess<NumericType, D>>::New(
        -1.0, 1.0, 100.0, Material::Mask);

    for (const auto &fluxEngine : fluxEngines) {
      runBenchmark<NumericType, D>(geometry, model, fluxEngine,
                                   "result_SP_" + util::toString(fluxEngine) +
                                       ".vtp");
    }
  }

  {
    IBEParameters<NumericType> ibeParams;
    ibeParams.cos4Yield.isDefined = true;
    ibeParams.cos4Yield.a1 = 1.075;
    ibeParams.cos4Yield.a2 = -1.55;
    ibeParams.cos4Yield.a3 = 0.65;
    ibeParams.materialPlaneWaferRate[Material::Si] = 2.0;
    ibeParams.materialPlaneWaferRate[Material::Mask] = 0.15;
    ibeParams.thetaRMin = 60;
    auto model = SmartPointer<IonBeamEtching<NumericType, D>>::New(ibeParams);

    for (const auto &fluxEngine : fluxEngines) {
      runBenchmark<NumericType, D>(geometry, model, fluxEngine,
                                   "result_IBE_" + util::toString(fluxEngine) +
                                       ".vtp");
    }
  }

  {
    std::function<NumericType(const std::vector<NumericType> &,
                              const Material &)>
        rateFunc =
            [](std::vector<NumericType> const &fluxes, const Material &m) {
              if (m == Material::Mask)
                return -0.1 * fluxes[0];
              return -fluxes[0];
            };
    NumericType thetaRMin = 60.;
    NumericType thetaRMax = 90.;
    NumericType minAngle = 80.;

    auto model = SmartPointer<MultiParticleProcess<NumericType, D>>::New();
    model->addIonParticle(100.0, thetaRMin, thetaRMax, minAngle);
    model->setRateFunction(rateFunc);

    for (const auto &fluxEngine : fluxEngines) {
      runBenchmark<NumericType, D>(geometry, model, fluxEngine,
                                   "result_MP_" + util::toString(fluxEngine) +
                                       ".vtp");
    }
  }
}
