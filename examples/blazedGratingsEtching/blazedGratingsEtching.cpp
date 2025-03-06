#include "blazedGratingsGeometry.hpp"

#include <models/psIonBeamEtching.hpp>
#include <psConstants.hpp>
#include <psProcess.hpp>
#include <psProcessParams.hpp>

using namespace viennaps;

int main(int argc, char **argv) {
  using NumericType = double;
  constexpr int D = 2;
  omp_set_num_threads(16);

  Logger::setLogLevel(LogLevel::INFO);

  // Parse the parameters
  utils::Parameters params;
  if (argc > 1) {
    params.readConfigFile(argv[1]);
  } else {
    std::cout << "Usage: " << argv[0] << " <config file>" << std::endl;
    return 1;
  }

  auto geometry = GenerateMask<NumericType, D>(
      params.get("bumpWidth"), params.get("bumpHeight"),
      params.get<int>("numBumps"), params.get("bumpDuty"),
      params.get("gridDelta"));
  geometry->saveSurfaceMesh("initial");

  AdvectionParameters<NumericType> advParams;
  advParams.integrationScheme =
      viennals::IntegrationSchemeEnum::LAX_FRIEDRICHS_2ND_ORDER;
  advParams.timeStepRatio = 0.25;

  RayTracingParameters<NumericType, D> rayTracingParams;
  rayTracingParams.raysPerPoint = params.get<unsigned>("raysPerPoint");
  rayTracingParams.smoothingNeighbors = 1;

  auto model = SmartPointer<IonBeamEtching<NumericType, D>>::New();
  const NumericType yieldFac = params.get("yieldFactor");

  auto &ibeParams = model->getParameters();
  ibeParams.materialPlaneWaferRate[Material::SiO2] = 1.0;
  ibeParams.materialPlaneWaferRate[Material::Mask] = 1. / 11.;
  ibeParams.exponent = params.get("exponent");
  ibeParams.meanEnergy = params.get("meanEnergy");
  ibeParams.yieldFunction = [yieldFac](NumericType theta) {
    const auto cosTheta = std::cos(theta);
    return (yieldFac * cosTheta - 1.55 * cosTheta * cosTheta +
            0.65 * cosTheta * cosTheta * cosTheta) /
           (yieldFac - 0.9);
  };

  { // ANSGM Etch
    const NumericType angle = params.get("phi1");
    const NumericType angleRad = constants::degToRad(angle);
    model->setPrimaryDirection({-std::sin(angleRad), -std::cos(angleRad), 0});
    ibeParams.tiltAngle = angle;

    Process<NumericType, D> process(geometry, model);
    process.setAdvectionParameters(advParams);
    process.setRayTracingParameters(rayTracingParams);

    process.setProcessDuration(params.get("ANSGM_Depth"));
    process.apply();
    geometry->saveSurfaceMesh("ANSGM_Etch");
  }

  geometry->removeTopLevelSet(); // remove mask
  geometry->saveSurfaceMesh("ANSGM");

  { // Blazed Gratings Etch
    const NumericType angle = params.get("phi2");
    const NumericType angleRad = constants::degToRad(angle);
    model->setPrimaryDirection({-std::sin(angleRad), -std::cos(angleRad), 0});
    ibeParams.tiltAngle = angle;

    Process<NumericType, D> process(geometry, model);
    process.setAdvectionParameters(advParams);
    process.setRayTracingParameters(rayTracingParams);

    for (int i = 1; i < 5; ++i) {
      process.setProcessDuration(params.get("etchTimeP" + std::to_string(i)));
      process.apply();
      geometry->saveSurfaceMesh("BlazedGratingsEtch_P" + std::to_string(i));
    }
  }
}