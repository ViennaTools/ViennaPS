#include <geometries/psMakeHole.hpp>
#include <models/psNeutralTransport.hpp>

#include <process/psProcess.hpp>
#include <psUtil.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <vector>

namespace ps = viennaps;

namespace {

template <typename NumericType>
void printBottomTransmissionProbabilityTriangles(
    ps::SmartPointer<viennals::Mesh<float>> triangleMesh,
    NumericType bottomRadius, NumericType topRadius, NumericType gridDelta,
    const std::string &fluxLabel) {
  if (triangleMesh == nullptr) {
    std::cout << "Triangle transmission probability: mesh unavailable "
                 "(triangle flux engine required)."
              << std::endl;
    return;
  }

  const auto flux = triangleMesh->getCellData().getScalarData(fluxLabel);
  if (flux == nullptr || flux->size() != triangleMesh->triangles.size()) {
    std::cout << "Triangle transmission probability: flux data unavailable."
              << std::endl;
    return;
  }

  const auto &nodes = triangleMesh->nodes;
  const auto &tris = triangleMesh->triangles;

  const float tol = static_cast<float>(gridDelta);
  const float bottomRMax = static_cast<float>(bottomRadius) + tol;
  const double topArea = M_PI * static_cast<double>(topRadius) *
                         static_cast<double>(topRadius) / 4.;

  struct TriangleInfo {
    float cx = 0.f;
    float cy = 0.f;
    float cz = 0.f;
    float radialPosition = 0.f;
    float normalZAbs = 0.f;
    float area = 0.f;
    float flux = 0.f;
  };

  std::vector<TriangleInfo> triangleInfos;
  triangleInfos.reserve(tris.size());

  float bottomZ = std::numeric_limits<float>::max();

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
    const float e1x = v1[0] - v0[0], e1y = v1[1] - v0[1], e1z = v1[2] - v0[2];
    const float e2x = v2[0] - v0[0], e2y = v2[1] - v0[1], e2z = v2[2] - v0[2];
    const float area =
        0.5f * std::sqrt((e1y * e2z - e1z * e2y) * (e1y * e2z - e1z * e2y) +
                         (e1z * e2x - e1x * e2z) * (e1z * e2x - e1x * e2z) +
                         (e1x * e2y - e1y * e2x) * (e1x * e2y - e1y * e2x));

    const float f = flux->at(i);
    const float radialPosition = std::sqrt(cx * cx + cy * cy);
    const float normalMagnitude = 2.f * area;
    const float normalZAbs =
        normalMagnitude > 0.f
            ? std::abs(e1x * e2y - e1y * e2x) / normalMagnitude
            : 0.f;

    triangleInfos.push_back({cx, cy, cz, radialPosition, normalZAbs, area, f});

    if (normalZAbs > 0.5f && radialPosition <= bottomRMax) {
      bottomZ = std::min(bottomZ, cz);
    }
  }

  if (bottomZ == std::numeric_limits<float>::max() || topArea <= 0.) {
    std::cout << "Triangle transmission probability: could not identify "
                 "bottom triangles or aperture area."
              << std::endl;
    return;
  }

  double bottomFluxIntegral = 0.;
  double bottomArea = 0.;
  std::size_t bottomCount = 0;

  for (const auto &info : triangleInfos) {
    if (std::abs(info.cz - bottomZ) <= tol && info.normalZAbs > 0.5f &&
        info.radialPosition <= bottomRMax) {
      bottomFluxIntegral += static_cast<double>(info.flux) * info.area;
      bottomArea += info.area;
      ++bottomCount;
    }
  }

  if (bottomArea == 0.) {
    std::cout << "Triangle transmission probability: could not identify "
                 "bottom etch-front triangles."
              << std::endl;
    return;
  }

  const double bottomDensity = bottomFluxIntegral / bottomArea;
  const double transmission = bottomFluxIntegral / topArea;

  std::cout << std::setprecision(6) << std::scientific
            << "Bottom transmission probability (triangles): " << transmission
            << " (bottom integral " << bottomFluxIntegral << ", aperture area "
            << topArea << ", bottom flux density " << bottomDensity << " from "
            << bottomCount << " triangles)" << std::endl;
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
  modelParams.desorptionMaterial = ps::MaterialMap::fromString(
      params.get<std::string>("desorptionMaterial"));
  modelParams.kEtch = params.get("kEtch");
  modelParams.surfaceSiteDensity = params.get("surfaceSiteDensity");
  modelParams.siliconDensity = params.get("siliconDensity");
  modelParams.coverageTimeStep = params.get("coverageTimeStep");
  modelParams.useSteadyStateCoverage =
      params.get<bool>("useSteadyStateCoverage");
  modelParams.surfaceDiffusionCoefficient =
      params.get("surfaceDiffusionCoefficient");
  modelParams.surfaceDiffusionMaterial = ps::MaterialMap::fromString(
      params.get<std::string>("surfaceDiffusionMaterial"));
  modelParams.surfaceDiffusionNeighborDistance =
      params.get("surfaceDiffusionNeighborDistance");
  modelParams.surfaceDiffusionSolverTolerance =
      params.get("surfaceDiffusionSolverTolerance");
  modelParams.surfaceDiffusionMaxIterations =
      params.get<unsigned>("surfaceDiffusionMaxIterations");
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
  const auto topRadius =
      params.get<NumericType>("holeRadius") +
      std::tan(params.get<NumericType>("maskTaperAngle") * M_PI / 180.) *
          params.get<NumericType>("maskHeight");
  printBottomTransmissionProbabilityTriangles(
      diagnosticProcess.getTriangleMesh(), params.get("holeRadius"), topRadius,
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
