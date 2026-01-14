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

namespace viennacore {

template <class T, int D>
void checkSurfaceHeight(SmartPointer<viennaps::Domain<T, D>> &domain,
                        const T height) {
  auto surfaceMesh = domain->getSurfaceMesh();
  for (const auto &node : surfaceMesh->nodes) {
    VC_TEST_ASSERT(std::abs(node[D - 1] - height) < 1e-1);
  }
}

template <class T, int D> void RunTest() {

  Logger::setLogLevel(LogLevel::WARNING);
  std::vector<viennaps::FluxEngineType> engineTypes = {
      viennaps::FluxEngineType::CPU_DISK,
      viennaps::FluxEngineType::CPU_TRIANGLE};
#ifdef VIENNACORE_COMPILE_GPU
  engineTypes.push_back(viennaps::FluxEngineType::GPU_DISK);
  engineTypes.push_back(viennaps::FluxEngineType::GPU_TRIANGLE);
  if constexpr (D == 2)
    engineTypes.push_back(viennaps::FluxEngineType::GPU_LINE);
#endif

  auto model =
      viennaps::SmartPointer<viennaps::SingleParticleProcess<T, D>>::New(
          1.0, 1.0, 1.0);

  for (const auto &engineType : engineTypes) {
    std::cout << "Testing flux engine: " << viennaps::util::toString(engineType)
              << " in " << D << "D" << std::endl;
    auto domain = viennaps::Domain<T, D>::New(1.0, 10.0, 10.0);

    viennaps::MakePlane<T, D>(domain, 0.0).apply();

    viennaps::RayTracingParameters rayParams;
    rayParams.rngSeed = 42;

    viennaps::Process<T, D> process(domain, model);
    process.setFluxEngineType(engineType);
    process.setProcessDuration(1.0);
    process.setParameters(rayParams);

    process.apply();

    domain->saveSurfaceMesh("fluxEngineTest_" +
                            std::to_string(static_cast<int>(D)) + "D_" +
                            viennaps::util::toString(engineType) + ".vtp");

    checkSurfaceHeight<T, D>(domain, 1.0);
  }
}

} // namespace viennacore

int main() { VC_RUN_ALL_TESTS }