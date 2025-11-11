#include <geometries/psMakeHole.hpp>
#include <models/psSF6O2Etching.hpp>
#include <models/psSingleParticleProcess.hpp>

#include <process/psProcess.hpp>
#include <psUtil.hpp>

using namespace viennaps;

int main(int argc, char *argv[]) {
  using NumericType = double;
  constexpr int D = 3;

  omp_set_num_threads(16);

  // set parameter units
  units::Length::setUnit(units::Length::MICROMETER);
  units::Time::setUnit(units::Time::MINUTE);
  Logger::setLogLevel(LogLevel::INFO);

  // geometry setup
  auto geometry = Domain<NumericType, D>::New(0.03, 1.2, 1.2);
  MakeHole<NumericType, D>(geometry, 0.175,
                           0.0, // holeDepth
                           0.0, // holeTaperAngle
                           1.2, 1.2, HoleShape::FULL)
      .apply();

  auto modelParams = SF6O2Etching<NumericType, D>::defaultParameters();
  modelParams.etchantFlux = 4.5e3;
  modelParams.passivationFlux = 8e2;
  modelParams.ionFlux = 10.0;
  auto model = SmartPointer<SF6O2Etching<NumericType, D>>::New(modelParams);

  // auto model = SmartPointer<SingleParticleProcess<NumericType, D>>::New(
  //     -1.0, 1.0, 100.0, Material::Mask);

  CoverageParameters coverageParams;
  coverageParams.tolerance = 1e-4;

  std::array<FluxEngineType, 4> fluxEngines = {
      FluxEngineType::GPU_TRIANGLE, FluxEngineType::GPU_DISK,
      FluxEngineType::CPU_TRIANGLE, FluxEngineType::CPU_DISK};

  Timer timer;
  for (const auto &fluxEngine : fluxEngines) {
    auto copy = Domain<NumericType, D>::New(geometry);

    Process<NumericType, D> process(copy, model);
    process.setProcessDuration(1.0);
    process.setParameters(coverageParams);
    process.setFluxEngineType(fluxEngine);

    timer.start();
    process.apply();
    timer.finish();

    std::cout << "Flux Engine: " << to_string(fluxEngine)
              << ", Time taken: " << timer.currentDuration / 1e9 << " s"
              << std::endl;
    copy->saveSurfaceMesh("result_PE_" + to_string(fluxEngine) + ".vtp");
  }
}
