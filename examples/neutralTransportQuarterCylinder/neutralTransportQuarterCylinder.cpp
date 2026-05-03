#include <geometries/psMakeHole.hpp>
#include <models/psNeutralTransport.hpp>

#include <process/psProcess.hpp>
#include <psUtil.hpp>

#include <cmath>
#include <iomanip>
#include <limits>

namespace ps = viennaps;

namespace {

template <typename NumericType>
void printBottomTransmissionProbabilityTriangles(
    ps::SmartPointer<viennals::Mesh<float>> triangleMesh,
    NumericType holeRadius, NumericType gridDelta,
    const std::string &fluxLabel) {
  if (triangleMesh == nullptr) {
    std::cout << "Triangle transmission probability: mesh unavailable "
                 "(GPU engine required)."
              << std::endl;
    return;
  }

  const auto flux =
      triangleMesh->getCellData().getScalarData(fluxLabel);
  if (flux == nullptr || flux->size() != triangleMesh->triangles.size()) {
    std::cout << "Triangle transmission probability: flux data unavailable."
              << std::endl;
    return;
  }

  const auto &nodes = triangleMesh->nodes;
  const auto &tris = triangleMesh->triangles;

  // find the z extents
  float bottomZ = std::numeric_limits<float>::max();
  float topZ = std::numeric_limits<float>::lowest();
  for (const auto &n : nodes) {
    bottomZ = std::min(bottomZ, n[2]);
    topZ = std::max(topZ, n[2]);
  }

  const float tol = static_cast<float>(gridDelta);
  const float rMax = static_cast<float>(holeRadius) + tol;

  struct Group {
    double fluxTimesArea = 0.;
    double area = 0.;
    std::size_t count = 0;
  } bottom, top;

  for (std::size_t i = 0; i < tris.size(); ++i) {
    const auto &t = tris[i];
    const auto &v0 = nodes[t[0]];
    const auto &v1 = nodes[t[1]];
    const auto &v2 = nodes[t[2]];

    // centroid
    const float cx = (v0[0] + v1[0] + v2[0]) / 3.f;
    const float cy = (v0[1] + v1[1] + v2[1]) / 3.f;
    const float cz = (v0[2] + v1[2] + v2[2]) / 3.f;

    // triangle area via cross product of edges
    const float e1x = v1[0] - v0[0], e1y = v1[1] - v0[1],
                e1z = v1[2] - v0[2];
    const float e2x = v2[0] - v0[0], e2y = v2[1] - v0[1],
                e2z = v2[2] - v0[2];
    const float area =
        0.5f * std::sqrt((e1y * e2z - e1z * e2y) * (e1y * e2z - e1z * e2y) +
                         (e1z * e2x - e1x * e2z) * (e1z * e2x - e1x * e2z) +
                         (e1x * e2y - e1y * e2x) * (e1x * e2y - e1y * e2x));

    const float f = flux->at(i);

    if (std::abs(cz - bottomZ) <= tol &&
        std::sqrt(cx * cx + cy * cy) <= rMax) {
      bottom.fluxTimesArea += static_cast<double>(f) * area;
      bottom.area += area;
      ++bottom.count;
    } else if (std::abs(cz - topZ) <= tol) {
      top.fluxTimesArea += static_cast<double>(f) * area;
      top.area += area;
      ++top.count;
    }
  }

  if (bottom.area == 0. || top.area == 0.) {
    std::cout << "Triangle transmission probability: could not identify "
                 "bottom or top triangles."
              << std::endl;
    return;
  }

  const double bottomDensity = bottom.fluxTimesArea / bottom.area;
  const double topDensity = top.fluxTimesArea / top.area;
  const double transmission = bottomDensity / topDensity;

  std::cout << std::setprecision(6) << std::scientific
            << "Bottom transmission probability (triangles): " << transmission
            << " (bottom flux density " << bottomDensity << " from "
            << bottom.count << " triangles, top flux density " << topDensity
            << " from " << top.count << " triangles)" << std::endl;
}

} // namespace

int main(int argc, char *argv[]) {
  using NumericType = double;
  constexpr int D = 3;

  ps::Logger::setLogLevel(ps::LogLevel::INFO);
  // omp_set_num_threads(16);

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
      geometry, params.get("holeRadius"),
      0.0, // start from a flat substrate; only open the mask
      0.0, // no taper in the substrate etch front
      params.get("maskHeight"), params.get("maskTaperAngle"),
      ps::HoleShape::QUARTER)
      .apply();

  ps::NeutralTransportParameters<NumericType> modelParams;
  modelParams.sourcePressure = params.get("sourcePressure");
  modelParams.sourceTemperature = params.get("sourceTemperature");
  modelParams.sourceMolecularMass = params.get("sourceMolecularMass");
  modelParams.incomingFlux = ps::molecularEffusionFlux(
      modelParams.sourcePressure, modelParams.sourceTemperature,
      modelParams.sourceMolecularMass);
  modelParams.zeroCoverageSticking = params.get("zeroCoverageSticking");
  modelParams.etchFrontSticking = params.get("etchFrontSticking");
  modelParams.desorptionRate = params.get("desorptionRate");
  modelParams.kEtch = params.get("kEtch");
  modelParams.surfaceSiteDensity = params.get("surfaceSiteDensity");
  modelParams.siliconDensity = params.get("siliconDensity");
  modelParams.coverageTimeStep = params.get("coverageTimeStep");
  modelParams.useSteadyStateCoverage =
      params.get<bool>("useSteadyStateCoverage");
  modelParams.surfaceDiffusionCoefficient =
      params.get("surfaceDiffusionCoefficient");
  modelParams.surfaceDiffusionRadius = params.get("surfaceDiffusionRadius");
  modelParams.surfaceDiffusionTolerance =
      params.get("surfaceDiffusionTolerance");
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

  ps::Process<NumericType, D> diagnosticProcess(geometry, model);
  diagnosticProcess.setParameters(coverageParams);
  diagnosticProcess.setParameters(rayTracingParams);
  diagnosticProcess.setFluxEngineType(
      ps::util::convertFluxEngineType(params.get<std::string>("fluxEngine")));
  diagnosticProcess.calculateFlux();
  printBottomTransmissionProbabilityTriangles(
      diagnosticProcess.getTriangleMesh(), params.get("holeRadius"),
      params.get("gridDelta"), modelParams.fluxLabel);

  // ps::Process<NumericType, D> process(geometry, model);
  // process.setProcessDuration(params.get("processTime"));
  // process.setParameters(coverageParams);
  // process.setParameters(rayTracingParams);
  // process.setParameters(advectionParams);
  // process.setFluxEngineType(
  //     ps::util::convertFluxEngineType(params.get<std::string>("fluxEngine")));

  // auto initialFile = params.get<std::string>("initialFile");
  // geometry->saveSurfaceMesh(initialFile);

  // process.apply();

  // auto outputFile = params.get<std::string>("outputFile");
  // geometry->saveSurfaceMesh(outputFile);

  return 0;
}
