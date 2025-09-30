#include <geometries/psMakeHole.hpp>
#include <models/psMultiParticleProcess.hpp>
#include <models/psgMultiParticleProcess.hpp>
#include <process/psProcess.hpp>

#include <vcContext.hpp>

using namespace viennaps;

int main(int argc, char **argv) {
  using NumericType = float;
  constexpr int D = 3;

  Logger::setLogLevel(LogLevel::INFO);
  omp_set_num_threads(16);

  NumericType gridDelta = 0.025;
  NumericType xExtent = 1.0;
  NumericType yExtent = 1.0;
  NumericType holeRadius = 0.175;
  NumericType maskHeight = 1.2;
  NumericType taperAngle = 1.193;

  NumericType processDuration = 0.5;
  NumericType exponent = 500.;
  AdvectionParameters advParams;
  advParams.integrationScheme = IntegrationScheme::ENGQUIST_OSHER_2ND_ORDER;
  RayTracingParameters rtParams;
  rtParams.raysPerPoint = 1000;

  // geometry setup
  auto geometry = Domain<NumericType, D>::New();
  MakeHole<NumericType, D>(geometry, gridDelta, xExtent, yExtent, holeRadius,
                           maskHeight, taperAngle,
                           0,     // base height
                           false, // periodic boundary
                           true,  // create mask
                           Material::Si, HoleShape::HALF)
      .apply();

  auto rateFunction = [](const std::vector<NumericType> &flux,
                         const Material &mat) -> NumericType {
    if (MaterialMap::isMaterial(mat, Material::Mask))
      return -flux[0] * 0.01;
    return -flux[0];
  };

  // CPU
  if constexpr (true) {
    auto model = SmartPointer<MultiParticleProcess<NumericType, D>>::New();
    model->addIonParticle(exponent, 80, 90, 75);
    model->setRateFunction(rateFunction);

    auto copy = Domain<NumericType, D>::New(geometry);

    Process<NumericType, D> process(copy, model, processDuration);
    process.setParameters(advParams);
    rtParams.smoothingNeighbors = 1.;
    process.setParameters(rtParams);
    process.apply();

    copy->saveSurfaceMesh("cpu_result.vtp", true);
  }

  rtParams.raysPerPoint *= 10; // increase rays for GPU to reduce noise
  {
    DeviceContext::createContext();

    auto model = SmartPointer<gpu::MultiParticleProcess<NumericType, D>>::New();
    model->setRateFunction(rateFunction);
    model->addIonParticle(exponent, 80, 90, 75);

    auto copy = Domain<NumericType, D>::New(geometry);

    Process<NumericType, D> process(copy, model, processDuration);
    process.setParameters(advParams);
    rtParams.smoothingNeighbors = 1.;
    process.setParameters(rtParams);
    process.setFluxEngineType(FluxEngineType::GPU_TRIANGLE);
    process.apply();

    copy->saveSurfaceMesh("gpu_result_trig.vtp", true);
  }

  {
    DeviceContext::createContext();

    auto model = SmartPointer<gpu::MultiParticleProcess<NumericType, D>>::New();
    model->setRateFunction(rateFunction);
    model->addIonParticle(exponent, 80, 90, 75);

    auto copy = Domain<NumericType, D>::New(geometry);

    Process<NumericType, D> process(copy, model, processDuration);
    process.setParameters(advParams);
    rtParams.smoothingNeighbors = 1.;
    process.setParameters(rtParams);
    process.setFluxEngineType(FluxEngineType::GPU_DISK);
    process.apply();

    copy->saveSurfaceMesh("gpu_result_disk.vtp", true);
  }
}