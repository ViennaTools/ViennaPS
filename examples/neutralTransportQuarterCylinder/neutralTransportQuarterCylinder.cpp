#include <geometries/psMakeHole.hpp>
#include <models/psNeutralTransport.hpp>

#include <process/psProcess.hpp>
#include <psUtil.hpp>

namespace ps = viennaps;

int main(int argc, char *argv[]) {
  using NumericType = double;
  constexpr int D = 3;

  ps::Logger::setLogLevel(ps::LogLevel::INFO);
  omp_set_num_threads(16);

  ps::util::Parameters params;
  if (argc > 1) {
    params.readConfigFile(argv[1]);
  } else {
    params.readConfigFile("config.txt");
    if (params.m.empty()) {
      std::cout << "No configuration file provided!" << std::endl;
      std::cout << "Usage: " << argv[0] << " <config file>" << std::endl;
      return 1;
    }
  }

  ps::units::Length::setUnit(params.get<std::string>("lengthUnit"));
  ps::units::Time::setUnit(params.get<std::string>("timeUnit"));

  auto geometry = ps::Domain<NumericType, D>::New(
      params.get("gridDelta"), params.get("xExtent"), params.get("yExtent"));

  ps::MakeHole<NumericType, D>(
      geometry, params.get("holeRadius"), params.get("holeDepth"),
      0.0, // no taper in the benchmark geometry
      params.get("maskHeight"), params.get("maskTaperAngle"),
      ps::HoleShape::QUARTER)
      .apply();

  ps::NeutralTransportParameters<NumericType> modelParams;
  modelParams.incomingFlux = params.get("incomingFlux");
  modelParams.zeroCoverageSticking = params.get("zeroCoverageSticking");
  modelParams.etchFrontSticking = params.get("etchFrontSticking");
  modelParams.desorptionRate = params.get("desorptionRate");
  modelParams.surfaceSiteDensity = params.get("surfaceSiteDensity");
  modelParams.coverageTimeStep = params.get("coverageTimeStep");
  modelParams.useSteadyStateCoverage =
      params.get<bool>("useSteadyStateCoverage");
  modelParams.surfaceDiffusionCoefficient =
      params.get("surfaceDiffusionCoefficient");
  modelParams.surfaceDiffusionRadius = params.get("surfaceDiffusionRadius");
  modelParams.surfaceDiffusionTolerance =
      params.get("surfaceDiffusionTolerance");
  modelParams.etchRate = params.get("etchRate");
  modelParams.sourceDistributionPower = params.get("sourceExponent");

  auto model =
      ps::SmartPointer<ps::NeutralTransport<NumericType, D>>::New(modelParams);

  ps::CoverageParameters coverageParams;
  coverageParams.maxIterations = params.get<unsigned>("coverageInitIterations");
  coverageParams.tolerance = params.get("coverageTolerance");

  ps::RayTracingParameters rayTracingParams;
  rayTracingParams.raysPerPoint = params.get<unsigned>("raysPerPoint");

  ps::AdvectionParameters advectionParams;
  advectionParams.spatialScheme =
      ps::util::convertSpatialScheme(params.get<std::string>("spatialScheme"));
  advectionParams.temporalScheme = ps::util::convertTemporalScheme(
      params.get<std::string>("temporalScheme"));
  advectionParams.calculateIntermediateVelocities =
      params.get<bool>("calculateIntermediateVelocities");

  ps::Process<NumericType, D> process(geometry, model);
  process.setProcessDuration(params.get("processTime"));
  process.setParameters(coverageParams);
  process.setParameters(rayTracingParams);
  process.setParameters(advectionParams);
  process.setFluxEngineType(
      ps::util::convertFluxEngineType(params.get<std::string>("fluxEngine")));

  process.apply();

  auto outputFile = params.get<std::string>("outputFile");
  geometry->saveSurfaceMesh(outputFile);

  return 0;
}
