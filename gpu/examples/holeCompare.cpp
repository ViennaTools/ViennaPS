#include <geometries/psMakeHole.hpp>
#include <models/psMultiParticleProcess.hpp>
#include <psProcess.hpp>

#include <vcContext.hpp>

#include <models/psgIonBeamEtching.hpp>
#include <psgProcess.hpp>

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

  NumericType processDuration = 2.;
  auto integrationScheme =
      viennals::IntegrationSchemeEnum::ENGQUIST_OSHER_2ND_ORDER;
  int raysPerPoint = 1000;

  NumericType exponent = 500.;

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
  if constexpr (false) {
    auto model = SmartPointer<MultiParticleProcess<NumericType, D>>::New();
    model->addIonParticle(exponent, 60, 90, 75);
    model->setRateFunction(rateFunction);

    auto copy = Domain<NumericType, D>::New(geometry);

    Process<NumericType, D> process(copy, model, processDuration);
    process.setIntegrationScheme(integrationScheme);
    process.setNumberOfRaysPerPoint(raysPerPoint);
    process.apply();

    copy->saveSurfaceMesh("cpu_result.vtp", true);
  }

  {
    Context context;
    context.create();

    auto model =
        SmartPointer<gpu::IonBeamEtching<NumericType, D>>::New(exponent);
    model->setRateFunction(rateFunction);

    auto copy = Domain<NumericType, D>::New(geometry);

    gpu::Process<NumericType, D> process(context, copy, model, processDuration);
    process.setIntegrationScheme(integrationScheme);
    process.setNumberOfRaysPerPoint(raysPerPoint * 10);
    process.apply();

    copy->saveSurfaceMesh("gpu_result.vtp", true);
  }
}