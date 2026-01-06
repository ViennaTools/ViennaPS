#include <viennaps.hpp>

using namespace viennaps;

int main(int argc, char **argv) {
  using NumericType = double;
  constexpr int D = 3;

  Logger::setLogLevel(LogLevel::ERROR);
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

  constexpr NumericType gridDelta = 0.01 * (1. + 1e-12);
  BoundaryType boundaryConds[D] = {BoundaryType::REFLECTIVE_BOUNDARY,
                                   BoundaryType::REFLECTIVE_BOUNDARY,
                                   BoundaryType::INFINITE_BOUNDARY};

  auto mask =
      SmartPointer<GDSGeometry<NumericType, D>>::New(gridDelta, boundaryConds);
  mask->setBoundaryPadding(0.1, 0.1);
  GDSReader<NumericType, D>(mask, params.get<std::string>("gdsFile")).apply();

  // geometry setup
  auto geometry = Domain<NumericType, D>::New();
  auto maskLS = mask->layerToLevelSet(0, 0.0, 0.18);
  geometry->insertNextLevelSetAsMaterial(maskLS, Material::Mask);
  MakePlane<NumericType, D>(geometry, 0.0, Material::Si, true).apply();

  auto modelParams = HBrO2Etching<NumericType, D>::defaultParameters();
  modelParams.ionFlux = params.get("ionFlux");
  modelParams.etchantFlux = params.get("etchantFlux");
  modelParams.passivationFlux = params.get("oxygenFlux");
  modelParams.Ions.meanEnergy = params.get("meanEnergy");
  modelParams.Ions.sigmaEnergy = params.get("sigmaEnergy");
  modelParams.Ions.exponent = params.get("ionExponent");
  modelParams.Ions.n_l = 200;
  auto model = SmartPointer<HBrO2Etching<NumericType, D>>::New(modelParams);

  // Advection parameters
  AdvectionParameters advectionParams;
  advectionParams.spatialScheme =
      util::convertSpatialScheme(params.get<std::string>("spatialScheme"));

  RayTracingParameters rayParams;
  rayParams.raysPerPoint = params.get<int>("raysPerPoint");

  CoverageParameters coverageParams;
  coverageParams.maxIterations = 10;

  // Process setup
  Process<NumericType, D> process(geometry, model, params.get("processTime"));
  process.setParameters(advectionParams);
  process.setParameters(rayParams);
  process.setParameters(coverageParams);

  // print initial surface
  geometry->saveSurfaceMesh("DRAM_Initial.vtp");

  const int numSteps = params.get("numSteps");
  for (int i = 0; i < numSteps; ++i) {
    process.apply();
    geometry->saveSurfaceMesh("DRAM_Etched_" + std::to_string(i + 1) + ".vtp");
  }

  geometry->saveHullMesh("DRAM_Final");

  return 0;
}
