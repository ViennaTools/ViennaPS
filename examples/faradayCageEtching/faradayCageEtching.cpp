#include <geometries/psMakeHole.hpp>
#include <models/psIonBeamEtching.hpp>
#include <psProcess.hpp>

int main(int argc, char *argv[]) {
  using NumericType = double;
  constexpr int D = 3;

  psLogger::setLogLevel(psLogLevel::INTERMEDIATE);
  omp_set_num_threads(16);

  // Parse the parameters
  psUtils::Parameters params;
  if (argc > 1) {
    params.readConfigFile(argv[1]);
  } else {
    std::cout << "Usage: " << argv[0] << " <config file>" << std::endl;
    return 1;
  }

  // geometry setup
  auto geometry = psSmartPointer<psDomain<NumericType, D>>::New();
  psMakeHole<NumericType, D>(
      geometry, params.get("gridDelta"), params.get("xExtent"),
      params.get("yExtent"), params.get("holeRadius"), params.get("maskHeight"),
      params.get("taperAngle"), 0. /* base height */,
      false /* periodic boundary */, true /*create mask*/, psMaterial::Si)
      .apply();

  // use pre-defined model IBE etching model
  auto model = psSmartPointer<psIonBeamEtching<NumericType, D>>::New(
      std::vector<psMaterial>{psMaterial::Mask});

  // process setup
  psProcess<NumericType, D> process;
  process.setDomain(geometry);
  process.setProcessModel(model);
  process.setIntegrationScheme(
      lsIntegrationSchemeEnum::LOCAL_LAX_FRIEDRICHS_1ST_ORDER);
  process.setNumberOfRaysPerPoint(params.get<int>("raysPerPoint"));
  process.setProcessDuration(params.get("processTime"));

  // print initial surface
  geometry->saveSurfaceMesh("initial.vtp");

  // run the process
  process.apply();

  // print final surface
  geometry->saveSurfaceMesh("final.vtp");
}
