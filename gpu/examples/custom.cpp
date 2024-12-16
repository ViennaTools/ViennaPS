#include <geometries/psMakeTrench.hpp>
#include <psUtils.hpp>

// #include <psProcess.hpp>
// #include <pscuProcess.hpp>
#include <curtIndexMap.hpp>
#include <curtTracer.hpp>
#include <pscuProcessPipelines.hpp>

using namespace viennaps;

int main(int argc, char **argv) {

  omp_set_num_threads(16);
  constexpr int D = 3;
  using NumericType = float;

  gpu::Context context;
  gpu::CreateContext(context);
  Logger::setLogLevel(LogLevel::DEBUG);

  const NumericType gridDelta = 1.;
  const NumericType extent = 100.;
  const NumericType trenchWidth = 15.;
  const NumericType spacing = 20.;
  const NumericType maskHeight = 40.;

  const NumericType time = 3.;
  const NumericType sticking = 1.0;
  const NumericType exponent = 1000;
  const int raysPerPoint = 3000;

  auto domain = SmartPointer<Domain<NumericType, D>>::New();
  MakeTrench<NumericType, D>(domain, gridDelta, extent, extent, trenchWidth,
                             maskHeight, 0., 0., false, true, Material::Si)
      .apply();
  domain->saveSurfaceMesh("trench_initial.vtp");

  gpu::Particle<NumericType> particle{.name = "SingleParticle",
                                      .sticking = sticking,
                                      .cosineExponent = exponent};
  particle.dataLabels.push_back("flux");

  gpu::Tracer<NumericType, D> rayTrace(context, domain);
  rayTrace.setPipeline("custom_generated_SingleParticlePipeline.cu");
  // rayTrace.setNumberOfRaysPerPoint(raysPerPoint);
  // rayTrace.setUseRandomSeed(false);
  // rayTrace.setPeriodicBoundary(false);
  // rayTrace.insertNextParticle(particle);
  // auto numRates = rayTrace.prepareParticlePrograms();
  // auto fluxesIndexMap = gpu::IndexMap(rayTrace.getParticles());

  // rayTrace.updateSurface(); // also creates mesh
  // auto numElements = rayTrace.getNumberOfElements();

  // rayTrace.apply();

  // auto mesh = rayTrace.getSurfaceMesh();
  // std::vector<NumericType> flux(numElements);
  // rayTrace.getFlux(flux.data(), 0, 0);

  // mesh->getCellData().insertNextScalarData(flux, "flux");

  // viennals::VTKWriter<NumericType>(mesh, "trench_flux.vtp").apply();
  return 0;
}