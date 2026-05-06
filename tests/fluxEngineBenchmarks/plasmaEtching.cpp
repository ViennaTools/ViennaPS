#include <geometries/psMakeHole.hpp>
#include <models/psSF6O2Etching.hpp>

#include <process/psProcess.hpp>
#include <psUtil.hpp>

using namespace viennaps;

int main(int argc, char *argv[]) {
  using NumericType = double;
  constexpr int D = 2;

  Logger::setLogLevel(LogLevel::WARNING);
  // set parameter units
  units::Length::setUnit("um");
  units::Time::setUnit("min");

  auto run = [&](FluxEngineType fluxEngine) {
    // geometry setup
    auto geometry = Domain<NumericType, D>::New(
        0.03, 1.0,
        BoundaryType::REFLECTIVE_BOUNDARY); // gridDelta, xExtent, boundary
    MakeHole<NumericType, D>(geometry, 0.175,
                             0.0, // holeDepth
                             0.0, // holeTaperAngle
                             1.2, 1.193, HoleShape::QUARTER)
        .apply();

    // use pre-defined model SF6O2 etching model
    auto modelParams = SF6O2Etching<NumericType, D>::defaultParameters();
    modelParams.beta_E[Material::Si] = 1.0;
    modelParams.beta_E[Material::Mask] = 1.0;
    modelParams.beta_P[Material::Si] = 1.0;
    modelParams.beta_P[Material::Mask] = 1.0;

    modelParams.ionFlux = 1.;
    modelParams.etchantFlux = 4.5e3;
    modelParams.passivationFlux = 8e2;
    modelParams.Ions.meanEnergy = 100;
    modelParams.Ions.sigmaEnergy = 10;
    modelParams.Ions.exponent = 1000;
    modelParams.Passivation.A_ie = 2.0;
    modelParams.Substrate.A_ie = 7;
    auto model = SmartPointer<SF6O2Etching<NumericType, D>>::New(modelParams);

    CoverageParameters coverageParams;
    coverageParams.tolerance = 1e-4;

    RayTracingParameters rayTracingParams;
    rayTracingParams.raysPerPoint = 10000;

    Process<NumericType, D> process(geometry, model);
    process.setProcessDuration(1.0);
    process.setParameters(coverageParams);
    process.setFluxEngineType(fluxEngine);
    process.setParameters(rayTracingParams);

    process.apply();

    geometry->saveSurfaceMesh("PlasmaEtching_" + util::toString(fluxEngine));
  };

  run(FluxEngineType::GPU_TRIANGLE);
  run(FluxEngineType::GPU_DISK);
  run(FluxEngineType::CPU_TRIANGLE);
  run(FluxEngineType::CPU_DISK);
}
