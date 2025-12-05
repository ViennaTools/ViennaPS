#include <process/psCPUDiskEngine.hpp>
#include <process/psCPUTriangleEngine.hpp>
#include <process/psGPUDiskEngine.hpp>
#include <process/psGPULineEngine.hpp>
#include <process/psGPUTriangleEngine.hpp>

#include <geometries/psMakePlane.hpp>
#include <models/psSingleParticleProcess.hpp>
#include <process/psProcess.hpp>
#include <psDomain.hpp>

#include <vcTestAsserts.hpp>

using namespace viennaps;

int main() {
  Logger::setLogLevel(LogLevel::WARNING);
  constexpr int D = 3;
  using T = double;

  std::vector<FluxEngineType> engineTypes = {FluxEngineType::CPU_DISK,
                                             FluxEngineType::CPU_TRIANGLE};
#ifdef VIENNACORE_COMPILE_GPU
  engineTypes.push_back(FluxEngineType::GPU_DISK);
  engineTypes.push_back(FluxEngineType::GPU_TRIANGLE);
#endif

  auto model = SmartPointer<SingleParticleProcess<T, D>>::New(1.0, 1.0, 1.0);

  for (const auto &engineType : engineTypes) {
    std::cout << "Testing rng seed: " << to_string(engineType) << " in " << D
              << "D" << std::endl;
    auto domain = Domain<T, D>::New(1.0, 10.0, 10.0);

    MakePlane<T, D>(domain, 0.0).apply();

    RayTracingParameters rayParams;
    rayParams.raysPerPoint = 100;
    rayParams.rngSeed = 42;
    rayParams.useRandomSeeds = false; // it is important to disable random seeds
                                      // here, also when rngSeed is set

    std::vector<T> flux1, flux2;

    {
      Process<T, D> process(domain, model);
      process.setFluxEngineType(engineType);
      process.setParameters(rayParams);

      auto flux = process.calculateFlux();
      flux1 = std::move(*flux->getCellData().getScalarData("particleFlux"));
    }

    {
      Process<T, D> process(domain, model);
      process.setFluxEngineType(engineType);
      process.setParameters(rayParams);

      auto flux = process.calculateFlux();
      flux2 = std::move(*flux->getCellData().getScalarData("particleFlux"));
    }

    VC_TEST_ASSERT(flux1.size() == flux2.size());
    for (size_t i = 0; i < flux1.size(); ++i)
      VC_TEST_ASSERT(flux1[i] == flux2[i]);
  }
}