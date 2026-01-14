#include <viennaps.hpp>

using namespace viennaps;

int main(int argc, char **argv) {

  Logger::setLogLevel(LogLevel::INFO);
  omp_set_num_threads(12);

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

  // set parameter units
  units::Length::setUnit(params.get<std::string>("lengthUnit"));
  units::Time::setUnit(params.get<std::string>("timeUnit"));

  constexpr double gridDelta = 0.01 * (1. + 1e-12);
  BoundaryType boundaryConds[3] = {BoundaryType::REFLECTIVE_BOUNDARY,
                                   BoundaryType::REFLECTIVE_BOUNDARY,
                                   BoundaryType::INFINITE_BOUNDARY};

  auto mask = SmartPointer<GDSGeometry_double_3>::New(gridDelta, boundaryConds);
  mask->setBoundaryPadding(0.1, 0.1);
  GDSReader_double_3(mask, params.get<std::string>("gdsFile")).apply();

  // geometry setup
  auto geometry = Domain_double_3::New();
  auto maskLS = mask->layerToLevelSet(0, 0.0, 0.18);
  geometry->insertNextLevelSetAsMaterial(maskLS, Material::Mask);
  MakePlane_double_3(geometry, 0.0, Material::Si, true).apply();

  auto modelParams = HBrO2Etching_double_3::defaultParameters();
  modelParams.ionFlux = params.get("ionFlux");
  modelParams.etchantFlux = params.get("etchantFlux");
  modelParams.passivationFlux = params.get("oxygenFlux");
  modelParams.Ions.meanEnergy = params.get("meanEnergy");
  modelParams.Ions.sigmaEnergy = params.get("sigmaEnergy");
  modelParams.Ions.exponent = params.get("ionExponent");
  modelParams.Ions.n_l = 200;
  modelParams.Substrate.B_sp = 0.75;
  auto model = SmartPointer<HBrO2Etching_double_3>::New(modelParams);

  // Advection parameters
  AdvectionParameters advectionParams;
  advectionParams.spatialScheme =
      util::convertSpatialScheme(params.get<std::string>("spatialScheme"));

  RayTracingParameters rayParams;
  rayParams.raysPerPoint = params.get<int>("raysPerPoint");

  const std::string fluxEngineStr = params.get<std::string>("fluxEngine");
  const auto fluxEngine = util::convertFluxEngineType(fluxEngineStr);

  CoverageParameters coverageParams;
  coverageParams.tolerance = 1e-5;

  // Process setup
  Process_double_3 process(geometry, model, params.get("processTime"));
  process.setParameters(advectionParams);
  process.setParameters(rayParams);
  process.setParameters(coverageParams);

  process.setFluxEngineType(fluxEngine);
  // print initial surface
  geometry->saveSurfaceMesh("DRAM_Initial_" + fluxEngineStr + ".vtp");

  const int numSteps = params.get("numSteps");
  for (int i = 0; i < numSteps; ++i) {
    process.apply();
    geometry->saveSurfaceMesh("DRAM_Etched_" + fluxEngineStr + "_" +
                              std::to_string(i + 1) + ".vtp");
  }

  geometry->saveHullMesh("DRAM_Final_" + fluxEngineStr);

  return 0;
}
