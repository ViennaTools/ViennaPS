#include <geometries/psMakeTrench.hpp>

#include <models/psgMultiParticleProcess.hpp>
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
  constexpr NumericType trenchHeight = 15.;
  constexpr NumericType maskHeight = 40.;

  constexpr NumericType time = 30.;
  constexpr NumericType sticking = 1.0;
  constexpr NumericType rate = 1.0;
  constexpr NumericType exponent = 100.;

  auto domain =
      SmartPointer<Domain<NumericType, D>>::New(gridDelta, extent, extent);
  MakeTrench<NumericType, D>(domain, trenchWidth, trenchHeight, 0, maskHeight)
      .apply();
  domain->saveSurfaceMesh("trench_initial.vtp");

  auto model = SmartPointer<gpu::MultiParticleProcess<NumericType, D>>::New();
  model->addIonParticle(exponent, 70, 90, 75);
  model->setRateFunction(
      [rate](const std::vector<NumericType> &flux, Material mat) {
        return mat == Material::Mask ? 0. : -flux[0] * rate;
      });

  RayTracingParameters<NumericType, D> rayTracingParams;
  rayTracingParams.smoothingNeighbors = 2;
  rayTracingParams.raysPerPoint = 3000;

  gpu::Process<NumericType, D> process(context, domain, model, time);
  process.setRayTracingParameters(rayTracingParams);
  process.disableRandomSeeds();

  process.apply();

  domain->saveSurfaceMesh("trench_etch.vtp");
}