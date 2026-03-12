#define VIENNACORE_FORCE_NOLOAD_CUDA

#include <viennaps.hpp>

#include <vcTestAsserts.hpp>

int main() {
  {
    auto context = viennaps::DeviceContext::createContext();
    VC_TEST_ASSERT(!context);
  }

  // Test that GPU engine throws expected exception when GPU support is
  // compiled, but no CUDA driver is available.
  {
    auto domain = viennaps::Domain<double, 2>::New(
        1., 10., viennaps::BoundaryType::REFLECTIVE_BOUNDARY);
    viennaps::MakePlane(domain).apply();

    viennaps::Process<double, 2> process;
    process.setDomain(domain);
    process.setProcessModel(viennaps::SmartPointer<
                            viennaps::SingleParticleProcess<double, 2>>::New());
    process.setFluxEngineType(viennaps::FluxEngineType::GPU_TRIANGLE);
    process.setProcessDuration(.1);

    try {
      process.apply();
    } catch (const std::exception &e) {
      std::cout << "Caught expected exception: " << e.what() << std::endl;
    }
  }

  // Test that AUTO engine selection falls back to CPU when GPU support is
  // compiled but no CUDA driver is available.
  {
    auto domain = viennaps::Domain<double, 2>::New(
        1., 10., viennaps::BoundaryType::REFLECTIVE_BOUNDARY);
    viennaps::MakePlane(domain).apply();

    viennaps::Process<double, 2> process;
    process.setDomain(domain);
    process.setProcessModel(viennaps::SmartPointer<
                            viennaps::SingleParticleProcess<double, 2>>::New());
    process.setFluxEngineType(viennaps::FluxEngineType::AUTO);
    process.setProcessDuration(.1);

    try {
      process.apply();
    } catch (const std::exception &e) {
      VC_TEST_ASSERT(false);
    }
  }
}