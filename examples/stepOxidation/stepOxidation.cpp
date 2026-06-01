#include <models/psOxidation.hpp>
#include <process/psProcess.hpp>
#include <psDomain.hpp>
#include <psUtil.hpp>

#include <lsBooleanOperation.hpp>
#include <lsGeometricAdvect.hpp>
#include <lsMakeGeometry.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <iostream>
#include <omp.h>
#include <stdexcept>
#include <string>

namespace ps = viennaps;
namespace ls = viennals;

using NumericType = double;

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

std::string lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return value;
}

std::string getString(const ps::util::Parameters &params, const char *key,
                      const std::string &fallback) {
  const auto it = params.m.find(key);
  return it == params.m.end() ? fallback : it->second;
}

ps::OxidantType parseOxidant(const std::string &value) {
  const auto n = lower(value);
  if (n == "wet" || n == "h2o") return ps::OxidantType::Wet;
  if (n == "dry" || n == "o2")  return ps::OxidantType::Dry;
  throw std::invalid_argument("Unknown oxidant '" + value +
                              "'. Use wet/H2O or dry/O2.");
}

ps::SiliconOrientation parseOrientation(const std::string &value) {
  const auto n = lower(value);
  if (n == "100" || n == "<100>" || n == "si100") return ps::SiliconOrientation::Si100;
  if (n == "111" || n == "<111>" || n == "si111") return ps::SiliconOrientation::Si111;
  if (n == "poly" || n == "polysi" || n == "poly-silicon")
    return ps::SiliconOrientation::PolySi;
  throw std::invalid_argument("Unknown orientation '" + value +
                              "'. Use 100, 111, or poly.");
}

// ---------------------------------------------------------------------------
// Geometry helpers (templated on D)
//
// Coordinate convention (both 2D and 3D):
//   dim 0 = X  — lateral step direction (REFLECTIVE boundary)
//   dim 1 = Y  — height / growth direction (INFINITE boundary)
//   dim 2 = Z  — depth / extrusion direction, 3D only (REFLECTIVE boundary)
//
// `timeStep` is a maximum internal oxidation step; the model automatically
// subcycles below it when the CFL limit is smaller.
// ---------------------------------------------------------------------------

template <int D>
ps::SmartPointer<ls::Domain<NumericType, D>> makeBoxLevelSet(
    const double *bounds,
    typename ls::Domain<NumericType, D>::BoundaryType *boundaryCons,
    NumericType gridDelta,
    const ls::VectorType<NumericType, D> &minCorner,
    const ls::VectorType<NumericType, D> &maxCorner) {
  auto levelSet =
      ls::Domain<NumericType, D>::New(bounds, boundaryCons, gridDelta);
  ls::MakeGeometry<NumericType, D> makeGeometry(
      levelSet, ls::Box<NumericType, D>::New(minCorner, maxCorner));
  // Ignore the Y boundary (growth, INFINITE); respect lateral boundaries.
  std::array<bool, D> ignoreBoundary{};
  ignoreBoundary[1] = true;
  makeGeometry.setIgnoreBoundaryConditions(ignoreBoundary);
  makeGeometry.apply();
  return levelSet;
}

// Builds the step geometry: a flat Si surface at y=leftTop for x<stepX,
// stepping up/down to y=rightTop for x>stepX.
//
// In 3D a second step wall is added at z=stepZ, forming a true 3D convex
// corner: the raised platform occupies only the quadrant x>stepX AND z>stepZ.
// The two step faces (the XY-plane wall and the ZY-plane wall) meet at the
// corner edge running along Y at (stepX, y, stepZ).
template <int D>
ps::SmartPointer<ls::Domain<NumericType, D>> makeStepLevelSet(
    const double *bounds,
    typename ls::Domain<NumericType, D>::BoundaryType *boundaryCons,
    NumericType gridDelta, NumericType xMax, NumericType leftTop,
    NumericType rightTop, NumericType stepX, NumericType zExtent = 0,
    NumericType stepZ = 0) {
  auto step = ls::Domain<NumericType, D>::New(bounds, boundaryCons, gridDelta);

  // Horizontal plane at y=leftTop — everything below is solid.
  ls::VectorType<NumericType, D> planeOrigin{};
  planeOrigin[1] = leftTop;
  ls::VectorType<NumericType, D> planeNormal{};
  planeNormal[1] = NumericType(1);
  ls::MakeGeometry<NumericType, D>(
      step, ls::Plane<NumericType, D>::New(planeOrigin, planeNormal))
      .apply();

  // Corner block: raised platform from y=leftTop to y=rightTop.
  // 2D: x in [stepX, xMax]
  // 3D: x in [stepX, xMax] AND z in [stepZ, zExtent] — one quadrant only.
  ls::VectorType<NumericType, D> rightMin{};
  rightMin[0] = stepX;
  rightMin[1] = leftTop;
  ls::VectorType<NumericType, D> rightMax{};
  rightMax[0] = xMax;
  rightMax[1] = rightTop;
  if constexpr (D == 3) {
    rightMin[2] = stepZ;
    rightMax[2] = zExtent;
  }
  auto rightBlock =
      makeBoxLevelSet<D>(bounds, boundaryCons, gridDelta, rightMin, rightMax);

  ls::BooleanOperation<NumericType, D>(step, rightBlock,
                                       ls::BooleanOperationEnum::UNION)
      .apply();
  return step;
}

// ---------------------------------------------------------------------------
// Simulation driver
// ---------------------------------------------------------------------------

template <int D>
void run(const ps::util::Parameters &params) {
  omp_set_num_threads(params.get<int>("numThreads"));
  ps::Logger::setLogLevel(ps::LogLevel::ERROR);

  const NumericType gridDelta      = params.get("gridDelta");
  const NumericType xExtent        = params.get("xExtent");
  const NumericType yMin           = params.get("yMin");
  const NumericType yMax           = params.get("yMax");
  const NumericType stepX          = params.get("stepX");
  const NumericType leftSiTop      = params.get("leftSiTop");
  const NumericType rightSiTop     = params.get("rightSiTop");
  const NumericType oxideThickness = params.get("oxideThickness");
  const NumericType oxidationTime  = params.get("oxidationTime");
  const NumericType timeStep       = params.get("timeStep");
  const NumericType temperature    = params.get("temperature");
  const NumericType pressure       = params.get("pressure");

  // For 3D: zExtent defaults to xExtent; stepZ defaults to stepX (symmetric corner).
  const NumericType zExtent = [&]() -> NumericType {
    if constexpr (D != 3) return NumericType(0);
    const auto it = params.m.find("zExtent");
    return (it == params.m.end()) ? xExtent : params.get("zExtent");
  }();
  const NumericType stepZ = [&]() -> NumericType {
    if constexpr (D != 3) return NumericType(0);
    const auto it = params.m.find("stepZ");
    return (it == params.m.end()) ? stepX : params.get("stepZ");
  }();

  const auto oxidant     = parseOxidant(getString(params, "oxidant", "wet"));
  const auto orientation = parseOrientation(getString(params, "orientation", "100"));
  const auto outputPrefix = getString(params, "outputPrefix", "ps_step_oxidation");

  // Domain bounds: [X, Y] for 2D; [X, Y, Z] for 3D.
  double bounds[2 * D];
  bounds[0] = -xExtent; bounds[1] = xExtent;
  bounds[2] = yMin;     bounds[3] = yMax;
  if constexpr (D == 3) { bounds[4] = -zExtent; bounds[5] = zExtent; }

  typename ls::Domain<NumericType, D>::BoundaryType boundaryCons[D];
  boundaryCons[0] = ls::Domain<NumericType, D>::BoundaryType::REFLECTIVE_BOUNDARY;
  boundaryCons[1] = ls::Domain<NumericType, D>::BoundaryType::INFINITE_BOUNDARY;
  if constexpr (D == 3) {
    boundaryCons[2] = ls::Domain<NumericType, D>::BoundaryType::REFLECTIVE_BOUNDARY;
  }

  auto siInterface =
      makeStepLevelSet<D>(bounds, boundaryCons, gridDelta, xExtent, leftSiTop,
                          rightSiTop, stepX, zExtent, stepZ);

  auto domain = ps::Domain<NumericType, D>::New();
  domain->insertNextLevelSetAsMaterial(siInterface, ps::Material::Si);

  if (oxideThickness > NumericType(0)) {
    auto ambientInterface =
        ls::Domain<NumericType, D>::New(siInterface);
    auto initialOxide =
        ps::SmartPointer<ls::SphereDistribution<viennahrle::CoordType, D>>::New(
            oxideThickness);
    ls::GeometricAdvect<NumericType, D>(ambientInterface, initialOxide).apply();
    domain->insertNextLevelSetAsMaterial(ambientInterface, ps::Material::SiO2,
                                         false);
  }
  auto model = ps::SmartPointer<ps::Oxidation<NumericType, D>>::New();
  model->setTemperature(temperature);
  model->setTime(oxidationTime);
  model->setTimeStep(timeStep);
  model->setOxidant(oxidant);
  model->setPressure(pressure);
  model->setOrientation(orientation);
  // Only override the native-oxide seed thickness when the caller
  // provides a pre-grown oxide. When oxideThickness == 0 there is no SiO2
  // layer in the domain, so the model seeds a default 2 nm native oxide;
  // setting the seed to 0 would leave no oxide volume for the solver.
  if (oxideThickness > NumericType(0))
    model->setInitialOxideThickness(oxideThickness);

  // maxGridPoints limits the Cartesian solve grid. Memory scales as N³ in 3D,
  // so either set a higher value here or coarsen gridDelta for 3D runs.
  {
    const auto it = params.m.find("maxGridPoints");
    if (it != params.m.end())
      model->setMaxGridPoints(static_cast<std::size_t>(std::stoull(it->second)));
  }

  model->saveSurfaceMesh(domain, outputPrefix + "_stack_initial.vtp");

  ps::Process<NumericType, D>(domain, model, NumericType(0)).apply();

  model->saveSurfaceMesh(domain, outputPrefix + "_stack_after.vtp");
  model->saveVolumeMesh(domain, outputPrefix + "_stack_after");
  

  std::cout << "Planar Deal-Grove estimate for " << oxidationTime
            << " hr oxidation at " << temperature << " C: "
            << model->estimatePlanarOxideThickness(oxideThickness)
            << " um oxide thickness." << std::endl;

  std::cout << "Wrote " << outputPrefix << "_stack_initial.vtp and "
            << outputPrefix << "_stack_after.vtp and "
            << outputPrefix << "_stack_after_volume.vtu" << std::endl;
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
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

  const int dimensions = std::stoi(getString(params, "dimensions", "2"));
  if (dimensions == 3)
    run<3>(params);
  else
    run<2>(params);

  return 0;
}
