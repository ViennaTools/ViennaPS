#include <geometries/psMakeTrench.hpp>
#include <models/psSingleParticleProcess.hpp>

#include <process/psProcess.hpp>
#include <psUtil.hpp>

using namespace viennaps;

int main(int argc, char *argv[]) {
  using NumericType = double;
  constexpr int D = 2;

#ifndef NDEBUG
  Logger::setLogLevel(LogLevel::INTERMEDIATE);
//   omp_set_num_threads(1);
#else
  Logger::setLogLevel(LogLevel::WARNING);
#endif

  auto run = [&](FluxEngineType fluxEngine) {
    auto geometry = Domain<NumericType, D>::New(
        0.14, 10, BoundaryType::REFLECTIVE_BOUNDARY);
    MakeTrench<NumericType, D>(geometry, 4.0, 20.0, 0.0, 0.0, 0.0, true)
        .apply();

    geometry->duplicateTopLevelSet(Material::SiO2);

    auto model =
        SmartPointer<SingleParticleProcess<NumericType, D>>::New(1.0, 0.01);
    model->setProcessName(util::toString(fluxEngine));

    RayTracingParameters rayTracingParams;
    rayTracingParams.raysPerPoint = 10000;

    Process<NumericType, D> process(geometry, model);
    process.setProcessDuration(5);
    process.setParameters(rayTracingParams);
    process.setFluxEngineType(fluxEngine);
    process.apply();
    // auto flux = process.calculateFlux();
    // viennals::VTKWriter<NumericType>(
    //     flux, "flux_" + util::toString(fluxEngine) + ".vtp")
    //     .apply();

    geometry->saveSurfaceMesh("spDeposition_" + util::toString(fluxEngine));
  };

  run(FluxEngineType::GPU_TRIANGLE);
  run(FluxEngineType::GPU_DISK);
  run(FluxEngineType::CPU_TRIANGLE);
  run(FluxEngineType::CPU_DISK);
}
