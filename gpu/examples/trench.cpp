#include <geometries/psMakeTrench.hpp>

#include <pscuProcess.hpp>
#include <pscuProcessPipelines.hpp>
#include <pscuSingleParticleProcess.hpp>

using namespace viennaps;

int main(int argc, char **argv) {

  omp_set_num_threads(16);
  constexpr int D = 3;
  using NumericType = float;

  gpu::Context context;
  gpu::CreateContext(context);
  Logger::setLogLevel(LogLevel::DEBUG);

  const NumericType gridDelta = 1.0;
  const NumericType extent = 100.;
  const NumericType trenchWidth = 15.;
  const NumericType maskHeight = 40.;

  const NumericType time = 10.;
  const NumericType sticking = .1;
  const NumericType rate = 1.0;
  const NumericType exponent = 1.;

  auto domain = SmartPointer<Domain<NumericType, D>>::New();

  MakeTrench<NumericType, D>(domain, gridDelta, extent, extent, trenchWidth,
                             maskHeight, 0., 0., false, false, Material::Si)
      .apply();
  domain->saveSurfaceMesh("trench_initial.vtp");

  auto model = SmartPointer<gpu::SingleParticleProcess<NumericType, D>>::New(
      rate, sticking, exponent);

  gpu::Process<NumericType, D> process(context, domain, model, time);
  process.setNumberOfRaysPerPoint(3000);

  domain->duplicateTopLevelSet(Material::SiO2);
  process.apply();

  domain->saveSurfaceMesh("trench_etched.vtp");
}