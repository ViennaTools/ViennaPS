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
// Geometry helpers
//
// Coordinate convention (both 2D and 3D):
//   dim 0 = X  — cross-section direction (REFLECTIVE boundary, trench centered at x=0)
//   dim 1 = Y  — height / growth direction (INFINITE boundary)
//   dim 2 = Z  — trench extrusion direction, 3D only (REFLECTIVE boundary)
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
  std::array<bool, D> ignoreBoundary{};
  ignoreBoundary[1] = true; // Y is INFINITE; no transformation to apply
  makeGeometry.setIgnoreBoundaryConditions(ignoreBoundary);
  makeGeometry.apply();
  return levelSet;
}

// Builds a trench geometry: flat Si substrate at y=0 with a rectangular
// trench centered at x=0, width trenchWidth, depth trenchDepth (into Si).
//
// In 3D the trench is extruded symmetrically along Z (slot geometry).
template <int D>
ps::SmartPointer<ls::Domain<NumericType, D>> makeTrenchLevelSet(
    const double *bounds,
    typename ls::Domain<NumericType, D>::BoundaryType *boundaryCons,
    NumericType gridDelta, NumericType trenchWidth, NumericType trenchDepth,
    NumericType zExtent) {
  // Flat substrate: solid below y = 0.
  auto siLS = ls::Domain<NumericType, D>::New(bounds, boundaryCons, gridDelta);
  ls::VectorType<NumericType, D> origin{}, normal{};
  normal[1] = NumericType(1);
  ls::MakeGeometry<NumericType, D>(
      siLS, ls::Plane<NumericType, D>::New(origin, normal))
      .apply();

  // Trench void box: the region to remove from the substrate.
  // Extends slightly above y=0 and below y=-trenchDepth for a clean boolean cut.
  ls::VectorType<NumericType, D> voidMin{}, voidMax{};
  voidMin[0] = -trenchWidth / NumericType(2);
  voidMin[1] = -trenchDepth - gridDelta;
  voidMax[0] =  trenchWidth / NumericType(2);
  voidMax[1] =  gridDelta;
  if constexpr (D == 3) { voidMin[2] = -zExtent; voidMax[2] = zExtent; }
  auto voidBox = makeBoxLevelSet<D>(bounds, boundaryCons, gridDelta, voidMin, voidMax);

  // Subtract the trench void from the substrate.
  ls::BooleanOperation<NumericType, D>(siLS, voidBox,
                                       ls::BooleanOperationEnum::RELATIVE_COMPLEMENT)
      .apply();
  return siLS;
}

// ---------------------------------------------------------------------------
// Simulation driver
// ---------------------------------------------------------------------------

template <int D>
void run(const ps::util::Parameters &params) {
  omp_set_num_threads(params.get<int>("numThreads"));
  ps::Logger::setLogLevel(ps::LogLevel::ERROR);

  const NumericType gridDelta     = params.get("gridDelta");
  const NumericType xExtent       = params.get("xExtent");
  const NumericType yMin          = params.get("yMin");
  const NumericType yMax          = params.get("yMax");
  const NumericType trenchWidth   = params.get("trenchWidth");
  const NumericType trenchDepth   = params.get("trenchDepth");
  const auto oxIt = params.m.find("oxideThickness");
  const NumericType oxideThickness =
      (oxIt != params.m.end()) ? params.get("oxideThickness") : NumericType(0);
  const NumericType oxidationTime = params.get("oxidationTime");
  const NumericType timeStep      = params.get("timeStep");
  const NumericType temperature   = params.get("temperature");
  const NumericType pressure      = params.get("pressure");

  // For 3D: zExtent defaults to xExtent (trench extrusion depth).
  const NumericType zExtent = [&]() -> NumericType {
    if constexpr (D != 3) return NumericType(0);
    const auto it = params.m.find("zExtent");
    return (it == params.m.end()) ? xExtent : params.get("zExtent");
  }();

  const auto oxidant      = parseOxidant(getString(params, "oxidant", "wet"));
  const auto orientation  = parseOrientation(getString(params, "orientation", "100"));
  const auto outputPrefix = getString(params, "outputPrefix", "ps_trench_oxidation");

  double bounds[2 * D];
  bounds[0] = -xExtent; bounds[1] = xExtent;
  bounds[2] = yMin;     bounds[3] = yMax;
  if constexpr (D == 3) { bounds[4] = -zExtent; bounds[5] = zExtent; }

  typename ls::Domain<NumericType, D>::BoundaryType boundaryCons[D];
  boundaryCons[0] = ls::Domain<NumericType, D>::BoundaryType::REFLECTIVE_BOUNDARY;
  boundaryCons[1] = ls::Domain<NumericType, D>::BoundaryType::INFINITE_BOUNDARY;
  if constexpr (D == 3)
    boundaryCons[2] = ls::Domain<NumericType, D>::BoundaryType::REFLECTIVE_BOUNDARY;

  auto siInterface = makeTrenchLevelSet<D>(bounds, boundaryCons, gridDelta,
                                           trenchWidth, trenchDepth, zExtent);
  auto domain = ps::Domain<NumericType, D>::New();
  domain->insertNextLevelSetAsMaterial(siInterface, ps::Material::Si);

  // Clamp the oxide seed to at least gridDelta so the Cartesian solve always
  // has resolvable nodes between the Si and SiO2 level sets.
  const NumericType seedThickness = std::max(oxideThickness, gridDelta);
  {
    auto ambientInterface = ls::Domain<NumericType, D>::New(siInterface);
    auto initialOxide =
        ps::SmartPointer<ls::SphereDistribution<viennahrle::CoordType, D>>::New(
            seedThickness);
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
  model->setInitialOxideThickness(seedThickness);

  {
    const auto it = params.m.find("maxGridPoints");
    if (it != params.m.end())
      model->setMaxGridPoints(
          static_cast<std::size_t>(std::stoull(it->second)));
  }

  model->saveSurfaceMesh(domain, outputPrefix + "_stack_initial.vtp");
  model->saveVolumeMesh(domain, outputPrefix + "_stack_initial");

  ps::Process<NumericType, D>(domain, model, NumericType(0)).apply();

  model->saveSurfaceMesh(domain, outputPrefix + "_stack_after.vtp");
  model->saveVolumeMesh(domain, outputPrefix + "_stack_after");

  std::cout << "Planar Deal-Grove estimate for " << oxidationTime
            << " hr oxidation at " << temperature << " C: "
            << model->estimatePlanarOxideThickness(seedThickness)
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
