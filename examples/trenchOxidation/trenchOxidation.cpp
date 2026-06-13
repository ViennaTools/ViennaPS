#include <models/psOxidation.hpp>
#include <process/psProcess.hpp>
#include <psDomain.hpp>
#include <psUtil.hpp>
#include <geometries/psMakeTrench.hpp>

#include <lsGeometricAdvect.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
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
  if (n == "110" || n == "<110>" || n == "si110") return ps::SiliconOrientation::Si110;
  if (n == "111" || n == "<111>" || n == "si111") return ps::SiliconOrientation::Si111;
  if (n == "poly" || n == "polysi" || n == "poly-silicon")
    return ps::SiliconOrientation::PolySi;
  throw std::invalid_argument("Unknown orientation '" + value +
                              "'. Use 100, 110, 111, or poly.");
}

// ---------------------------------------------------------------------------
// Simulation driver
//
// Coordinate convention:
//   2D: X = lateral (REFLECTIVE), Y = growth (INFINITE)
//   3D: X = lateral (REFLECTIVE), Y = trench extrusion (REFLECTIVE), Z = growth (INFINITE)
//       yMin/yMax refer to the growth axis (Z in 3D); zExtent is the Y extrusion range.
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
  const NumericType temperature   = params.get("temperature");
  const NumericType pressure      = params.get("pressure");

  const NumericType zExtent = [&]() -> NumericType {
    if constexpr (D != 3) return NumericType(0);
    const auto it = params.m.find("zExtent");
    return (it == params.m.end()) ? xExtent : params.get("zExtent");
  }();

  const auto oxidant      = parseOxidant(getString(params, "oxidant", "wet"));
  const auto orientation  = parseOrientation(getString(params, "orientation", "100"));
  const auto outputPrefix = getString(params, "outputPrefix", "ps_trench_oxidation");

  using BoundaryType = typename ls::Domain<NumericType, D>::BoundaryType;
  double bounds[2 * D];
  BoundaryType boundaryCons[D];

  bounds[0] = -xExtent; bounds[1] = xExtent;
  if constexpr (D == 2) {
    bounds[2] = yMin; bounds[3] = yMax;
    boundaryCons[0] = BoundaryType::REFLECTIVE_BOUNDARY;
    boundaryCons[1] = BoundaryType::INFINITE_BOUNDARY;
  } else {
    // Y = trench extrusion (REFLECTIVE), Z = growth (INFINITE)
    bounds[2] = -zExtent; bounds[3] = zExtent;
    bounds[4] = yMin;     bounds[5] = yMax;
    boundaryCons[0] = BoundaryType::REFLECTIVE_BOUNDARY;
    boundaryCons[1] = BoundaryType::REFLECTIVE_BOUNDARY;
    boundaryCons[2] = BoundaryType::INFINITE_BOUNDARY;
  }

  auto domain = ps::Domain<NumericType, D>::New(bounds, boundaryCons, gridDelta);
  ps::MakeTrench<NumericType, D>(domain, trenchWidth, trenchDepth).apply();

  // Clamp the oxide seed to at least gridDelta so the Cartesian solve always
  // has resolvable nodes between the Si and SiO2 level sets.
  const NumericType seedThickness = std::max(oxideThickness, gridDelta);
  {
    auto ambientInterface = ls::Domain<NumericType, D>::New(domain->getLevelSets().back());
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
  model->setOxidant(oxidant);
  model->setPressure(pressure);
  model->setOrientation(orientation);
  model->setInitialOxideThickness(seedThickness);

  {
    const auto useGpu = lower(getString(params, "useGpu", "auto"));
    if      (useGpu == "gpu") model->setGpuMode(ps::GpuMode::Gpu);
    else if (useGpu == "cpu") model->setGpuMode(ps::GpuMode::Cpu);
    const auto prec = lower(getString(params, "gpuPreconditioner", "jacobi"));
    if (prec == "ilu0") model->setGpuPreconditioner(ps::GpuPreconditioner::ILU0);
  }

  {
    const auto it = params.m.find("maxGridPoints");
    if (it != params.m.end())
      model->setMaxGridPoints(
          static_cast<std::size_t>(std::stoull(it->second)));
  }

  model->saveSurfaceMesh(domain, outputPrefix + "_stack_initial.vtp");
  model->saveVolumeMesh(domain, outputPrefix + "_stack_initial");

  const auto t0 = std::chrono::steady_clock::now();
  ps::Process<NumericType, D>(domain, model, NumericType(0)).apply();
  const double elapsedSim =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  model->saveSurfaceMesh(domain, outputPrefix + "_stack_after.vtp");
  model->saveVolumeMesh(domain, outputPrefix + "_stack_after");

  std::cout << "Simulation time: " << elapsedSim << " s\n";
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
