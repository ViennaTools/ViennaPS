#include <geometries/psMakePlane.hpp>
#include <geometries/psMakeTrench.hpp>

#include <models/psgSingleParticleProcess.hpp>
#include <psgProcess.hpp>

using namespace viennaps;

int main(int argc, char **argv) {

  omp_set_num_threads(16);
  constexpr int D = 3;
  using NumericType = double;
  Logger::setLogLevel(LogLevel::INFO);

  Context context;
  context.create();

  constexpr NumericType gridDelta = 1.0;
  constexpr NumericType extent = 50.;
  constexpr NumericType trenchWidth = 15.;
  constexpr NumericType maskHeight = 40.;

  constexpr NumericType time = 30.;
  constexpr NumericType sticking = 1.0;
  constexpr NumericType rate = 1.0;
  constexpr NumericType exponent = 1.;

  auto domain = SmartPointer<Domain<NumericType, D>>::New();
  MakeTrench<NumericType, D>(domain, gridDelta, extent, extent, trenchWidth,
                             maskHeight, 0., 0., false, false, Material::Si)
      .apply();
  domain->saveSurfaceMesh("trench_initial.vtp");

  auto model = SmartPointer<gpu::SingleParticleProcess<NumericType, D>>::New(
      rate, sticking, exponent);

  gpu::Process<NumericType, D> process(context, domain, model, time);
  process.setNumberOfRaysPerPoint(3000);
  process.disableRandomSeeds();

  domain->duplicateTopLevelSet(Material::SiO2);
  process.apply();

  domain->saveSurfaceMesh("trench_depo.vtp");
}