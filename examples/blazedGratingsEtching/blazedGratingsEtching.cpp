#include "blazedGratingsGeometry.hpp"

#include <models/psIonBeamEtching.hpp>
#include <process/psProcess.hpp>
#include <psConstants.hpp>

using namespace viennaps;

int main(int argc, char **argv) {
  omp_set_num_threads(16);
  Logger::setLogLevel(LogLevel::DEBUG);

  // Parse the parameters
  util::Parameters params;
  if (argc > 1) {
    params.readConfigFile(argv[1]);
  } else {
    // Try default config file
    params.readConfigFile("config.txt");
    if (params.m.empty()) {
      std::cout << "No configuration file provided!" << std::endl;
      std::cout << "Usage: " << argv[0] << " <config file>" << std::endl;
      return 1;
    }
  }

  auto geometry =
      GenerateMask<double, 2>(params.get("bumpWidth"), params.get("bumpHeight"),
                              params.get<int>("numBumps"),
                              params.get("bumpDuty"), params.get("gridDelta"));
  geometry->saveSurfaceMesh("initial");

  AdvectionParameters advParams;
  advParams.spatialScheme =
      viennals::SpatialSchemeEnum::LAX_FRIEDRICHS_2ND_ORDER;
  advParams.timeStepRatio = 0.25;

  RayTracingParameters rayTracingParams;
  rayTracingParams.raysPerPoint = params.get<unsigned>("raysPerPoint");
  rayTracingParams.smoothingNeighbors = 1;

  const double yieldFac = params.get("yieldFactor");

  IBEParameters<double> ibeParams;
  ibeParams.materialPlaneWaferRate[Material::SiO2] = 1.0;
  ibeParams.materialPlaneWaferRate[Material::Mask] = 1. / 11.;
  ibeParams.exponent = params.get("exponent");
  ibeParams.meanEnergy = params.get("meanEnergy");
  ibeParams.cos4Yield.isDefined = true;
  ibeParams.cos4Yield.a1 = yieldFac;
  ibeParams.cos4Yield.a2 = -1.55;
  ibeParams.cos4Yield.a3 = 0.65;

  // ANSGM Etch
  {
    double angle = params.get("phi1");
    double angleRad = constants::degToRad(angle);
    ibeParams.tiltAngle = angle;
    auto model = SmartPointer<IonBeamEtching_double_2>::New(ibeParams);
    model->setPrimaryDirection(
        Vec3Dd{-std::sin(angleRad), -std::cos(angleRad), 0});
    model->setProcessName("ANSGM_Etch");

    Process_double_2(geometry, model, params.get("ANSGM_Depth"), advParams,
                     rayTracingParams)
        .apply();
    geometry->saveSurfaceMesh("ANSGM_Etch");
  }

  geometry->removeTopLevelSet(); // remove mask
  geometry->saveSurfaceMesh("ANSGM");

  // Blazed Gratings Etch
  {
    double angle = params.get("phi2");
    double angleRad = constants::degToRad(angle);
    ibeParams.tiltAngle = angle;
    auto model = SmartPointer<IonBeamEtching_double_2>::New(ibeParams);
    model->setPrimaryDirection(
        Vec3Dd{-std::sin(angleRad), -std::cos(angleRad), 0});
    model->setProcessName("BlazedGratings_Etch");

    Process_double_2 process(geometry, model);
    process.setParameters(advParams);
    process.setParameters(rayTracingParams);

    for (int i = 1; i < 5; ++i) {
      process.setProcessDuration(params.get("etchTimeP" + std::to_string(i)));
      process.apply();
      geometry->saveSurfaceMesh("BlazedGratingsEtch_P" + std::to_string(i));
    }
  }
}
