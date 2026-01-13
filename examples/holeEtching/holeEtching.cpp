#include <geometries/psMakeHole.hpp>
#include <models/psSF6O2Etching.hpp>

#include <process/psProcess.hpp>
#include <psUtil.hpp>

using namespace viennaps;

int main(int argc, char *argv[]) {
  using NumericType = double;
  constexpr int D = 3;

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

  // set parameter units
  units::Length::setUnit(params.get<std::string>("lengthUnit"));
  units::Time::setUnit(params.get<std::string>("timeUnit"));

  // geometry setup
  auto geometry = Domain<NumericType, D>::New(
      params.get("gridDelta"), params.get("xExtent"), params.get("yExtent"));
  MakeHole<NumericType, D>(geometry, params.get("holeRadius"),
                           0.0, // holeDepth
                           0.0, // holeTaperAngle
                           params.get("maskHeight"), params.get("taperAngle"),
                           HoleShape::HALF)
      .apply();

  // use pre-defined model SF6O2 etching model
  auto modelParams = SF6O2Etching<NumericType, D>::defaultParameters();
  modelParams.ionFlux = params.get("ionFlux");
  modelParams.etchantFlux = params.get("etchantFlux");
  modelParams.passivationFlux = params.get("oxygenFlux");
  modelParams.Ions.meanEnergy = params.get("meanEnergy");
  modelParams.Ions.sigmaEnergy = params.get("sigmaEnergy");
  modelParams.Ions.exponent = params.get("ionExponent");
  modelParams.Passivation.A_ie = params.get("A_O");
  modelParams.Substrate.A_ie = params.get("A_Si");
  modelParams.etchStopDepth = params.get("etchStopDepth");
  auto model = SmartPointer<SF6O2Etching<NumericType, D>>::New(modelParams);

  CoverageParameters coverageParams;
  coverageParams.tolerance = 1e-4;

  RayTracingParameters rayTracingParams;
  rayTracingParams.raysPerPoint = params.get<unsigned>("raysPerPoint");

  AdvectionParameters advectionParams;
  advectionParams.spatialScheme =
      util::convertSpatialScheme(params.get<std::string>("spatialScheme"));
  advectionParams.temporalScheme =
      util::convertTemporalScheme(params.get<std::string>("temporalScheme"));

  // process setup
  Process<NumericType, D> process(geometry, model);
  process.setProcessDuration(params.get("processTime"));
  process.setParameters(coverageParams);
  process.setParameters(rayTracingParams);
  process.setParameters(advectionParams);

  // print initial surface
  geometry->saveSurfaceMesh("initial.vtp");

  // run the process
  process.apply();

  // print final surface
  geometry->saveSurfaceMesh(params.get<std::string>("outputFile"), true, 0.01,
                            true);
}
