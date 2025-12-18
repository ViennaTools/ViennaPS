#include "blazedGratingsGeometry.hpp"

#include <models/psIonBeamEtching.hpp>
#include <process/psProcess.hpp>
#include <psConstants.hpp>

using namespace viennaps;

int main(int argc, char **argv) {
  using NumericType = double;
  constexpr int D = 2;
  omp_set_num_threads(16);

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

  auto geometry = GenerateMask<NumericType, D>(
      params.get("bumpWidth"), params.get("bumpHeight"),
      params.get<int>("numBumps"), params.get("bumpDuty"),
      params.get("gridDelta"));
  geometry->saveSurfaceMesh("initial");

  AdvectionParameters advParams;
  advParams.discretizationScheme =
      viennals::DiscretizationSchemeEnum::LAX_FRIEDRICHS_2ND_ORDER;
  advParams.timeStepRatio = 0.25;

  RayTracingParameters rayTracingParams;
  rayTracingParams.raysPerPoint = params.get<unsigned>("raysPerPoint");
  rayTracingParams.smoothingNeighbors = 1;

  const NumericType yieldFac = params.get("yieldFactor");

  IBEParameters<NumericType> ibeParams;
  ibeParams.materialPlaneWaferRate[Material::SiO2] = 1.0;
  ibeParams.materialPlaneWaferRate[Material::Mask] = 1. / 11.;
  ibeParams.exponent = params.get("exponent");
  ibeParams.meanEnergy = params.get("meanEnergy");
  ibeParams.cos4Yield.isDefined = true;
  ibeParams.cos4Yield.a1 = yieldFac;
  ibeParams.cos4Yield.a2 = -1.55;
  ibeParams.cos4Yield.a3 = 0.65;
  auto model = SmartPointer<IonBeamEtching<NumericType, D>>::New(ibeParams);

  Process<NumericType, D> process(geometry, model);
  process.setParameters(advParams);
  process.setParameters(rayTracingParams);

  // ANSGM Etch
  NumericType angle = params.get("phi1");
  NumericType angleRad = constants::degToRad(angle);
  model->setPrimaryDirection(
      Vec3D<NumericType>{-std::sin(angleRad), -std::cos(angleRad), 0});
  ibeParams.tiltAngle = angle;

  process.setProcessDuration(params.get("ANSGM_Depth"));
  process.apply();
  geometry->saveSurfaceMesh("ANSGM_Etch");

  geometry->removeTopLevelSet(); // remove mask
  geometry->saveSurfaceMesh("ANSGM");

  // Blazed Gratings Etch
  angle = params.get("phi2");
  angleRad = constants::degToRad(angle);
  model->setPrimaryDirection(
      Vec3D<NumericType>{-std::sin(angleRad), -std::cos(angleRad), 0});
  ibeParams.tiltAngle = angle;

  for (int i = 1; i < 5; ++i) {
    process.setProcessDuration(params.get("etchTimeP" + std::to_string(i)));
    process.apply();
    geometry->saveSurfaceMesh("BlazedGratingsEtch_P" + std::to_string(i));
  }
}
