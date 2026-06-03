// LOCOS (Local Oxidation of Silicon) process example using psOxidation.
//
// Geometry:
//   - Si substrate (plane at y = 0)
//   - Pad oxide (SiO2, thickness padOxideThickness) grown on Si before masking
//   - Si3N4 mask box (covering x < maskEdge, sitting on top of pad oxide)
//
// The Oxidation model auto-detects the Si3N4 material and activates LOCOS
// physics: mask bending + constrained-ambient advection, producing the
// characteristic bird's-beak oxide profile.
//
// Config keys (lengths in µm, time in hours, pressure in atm):
//   numThreads, gridDelta, xExtent, yMin, yMax,
//   padOxideThickness, maskThickness, maskEdge,
//   oxidationTime, timeStep, temperature, pressure, oxidant, orientation,
//   maxGridPoints, outputPrefix
//
// `timeStep` controls the output cadence and the maximum oxidation substep.
// The model automatically uses smaller CFL-limited physics steps if needed.

#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <sstream>
#include <string>

#include <lsBooleanOperation.hpp>
#include <lsDomain.hpp>
#include <lsGeometricAdvect.hpp>
#include <lsMakeGeometry.hpp>

#include <models/psOxidation.hpp>
#include <process/psProcess.hpp>
#include <psDomain.hpp>

namespace ls = viennals;
namespace ps = viennaps;

using NumericType = double;
constexpr int D = 2;

// ── Geometry helpers ─────────────────────────────────────────────────────────

using LevelSet = ls::SmartPointer<ls::Domain<NumericType, D>>;
using BoundaryType = ls::Domain<NumericType, D>::BoundaryType;

LevelSet makePlane(const double *bounds, BoundaryType *bc,
                   NumericType gridDelta, NumericType y) {
  auto ls = ls::Domain<NumericType, D>::New(bounds, bc, gridDelta);
  const ls::VectorType<NumericType, D> origin{0., y};
  const ls::VectorType<NumericType, D> normal{0., 1.};
  ls::MakeGeometry<NumericType, D>(ls,
                                   ls::Plane<NumericType, D>::New(origin, normal))
      .apply();
  return ls;
}

LevelSet makeMask(const double *bounds, BoundaryType *bc, NumericType gridDelta,
                  NumericType xMin, NumericType xMax, NumericType yMin,
                  NumericType yMax) {
  auto mask = ls::Domain<NumericType, D>::New(bounds, bc, gridDelta);
  const ls::VectorType<NumericType, D> minCorner{xMin, yMin};
  const ls::VectorType<NumericType, D> maxCorner{xMax, yMax};
  ls::MakeGeometry<NumericType, D> geom(
      mask, ls::Box<NumericType, D>::New(minCorner, maxCorner));
  geom.setIgnoreBoundaryConditions(std::array<bool, D>{false, true});
  geom.apply();
  return mask;
}

// ── Config parser ────────────────────────────────────────────────────────────

struct Config {
  int numThreads = 4;
  NumericType gridDelta = 0.05;
  NumericType xExtent = 4.;
  NumericType yMin = -1.;
  NumericType yMax = 2.;
  NumericType padOxideThickness = 0.05;
  NumericType maskThickness = 0.2;
  NumericType maskEdge = 0.;
  NumericType oxidationTime = 0.35;
  NumericType timeStep = 0.1;
  NumericType temperature = 1000.;
  NumericType pressure = 1.;
  std::string oxidant = "wet";
  std::string orientation = "100";
  std::size_t maxGridPoints = 5000000;
  std::string outputPrefix = "ps_locos";
  int mechanicsIterations = 2;
  NumericType mechanicsTolerance = 1e-7;
  int pressureIterations = 500;
  NumericType pressureTolerance = 1e-3;  // 1e-8 is unachievable with Jacobi; 1e-3 works with ILU(0)
  int stokesIterations = 100;
  NumericType stokesTolerance = 1e-3;
  int couplingIterations = 8;
  NumericType couplingTolerance = 1e-6;
  int maskCouplingIterations = 8;
  NumericType maskCouplingTolerance = 0.02;
  // "debug", "timing", "intermediate", "info" (default), "warning", "error"
  std::string logLevel = "info";
};

Config parseConfig(const std::string &filename) {
  Config cfg;
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Warning: could not open " << filename
              << ", using defaults.\n";
    return cfg;
  }
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#')
      continue;
    const auto eq = line.find('=');
    if (eq == std::string::npos)
      continue;
    const auto key = line.substr(0, eq);
    const auto val = line.substr(eq + 1);
    if (key == "numThreads")        cfg.numThreads = std::stoi(val);
    else if (key == "gridDelta")    cfg.gridDelta = std::stod(val);
    else if (key == "xExtent")      cfg.xExtent = std::stod(val);
    else if (key == "yMin")         cfg.yMin = std::stod(val);
    else if (key == "yMax")         cfg.yMax = std::stod(val);
    else if (key == "padOxideThickness") cfg.padOxideThickness = std::stod(val);
    else if (key == "maskThickness")cfg.maskThickness = std::stod(val);
    else if (key == "maskEdge")     cfg.maskEdge = std::stod(val);
    else if (key == "oxidationTime")cfg.oxidationTime = std::stod(val);
    else if (key == "timeStep")     cfg.timeStep = std::stod(val);
    else if (key == "temperature")  cfg.temperature = std::stod(val);
    else if (key == "pressure")     cfg.pressure = std::stod(val);
    else if (key == "oxidant")      cfg.oxidant = val;
    else if (key == "orientation")  cfg.orientation = val;
    else if (key == "maxGridPoints")cfg.maxGridPoints = std::stoull(val);
    else if (key == "outputPrefix") cfg.outputPrefix = val;
    else if (key == "mechanicsIterations")  cfg.mechanicsIterations  = std::stoi(val);
    else if (key == "mechanicsTolerance")   cfg.mechanicsTolerance   = std::stod(val);
    else if (key == "pressureIterations")   cfg.pressureIterations   = std::stoi(val);
    else if (key == "pressureTolerance")    cfg.pressureTolerance    = std::stod(val);
    else if (key == "stokesIterations") cfg.stokesIterations = std::stoi(val);
    else if (key == "stokesTolerance")  cfg.stokesTolerance  = std::stod(val);
    else if (key == "couplingIterations") cfg.couplingIterations = std::stoi(val);
    else if (key == "couplingTolerance") cfg.couplingTolerance = std::stod(val);
    else if (key == "maskCouplingIterations") cfg.maskCouplingIterations = std::stoi(val);
    else if (key == "maskCouplingTolerance") cfg.maskCouplingTolerance = std::stod(val);
    else if (key == "logLevel")              cfg.logLevel = val;
  }
  return cfg;
}

// ── Main ─────────────────────────────────────────────────────────────────────

int main() {
  const auto cfg = parseConfig("config.txt");

  // Apply log level from config (controls verbosity including timing output).
  {
    const auto &lv = cfg.logLevel;
    if      (lv == "debug")        ps::Logger::setLogLevel(ps::LogLevel::DEBUG);
    else if (lv == "timing")       ps::Logger::setLogLevel(ps::LogLevel::TIMING);
    else if (lv == "intermediate") ps::Logger::setLogLevel(ps::LogLevel::INTERMEDIATE);
    else if (lv == "info")         ps::Logger::setLogLevel(ps::LogLevel::INFO);
    else if (lv == "warning")      ps::Logger::setLogLevel(ps::LogLevel::WARNING);
    else if (lv == "error")        ps::Logger::setLogLevel(ps::LogLevel::ERROR);
    else                           ps::Logger::setLogLevel(ps::LogLevel::INFO);
  }
  omp_set_num_threads(cfg.numThreads);

  const NumericType maskContactEpsilon = 1.e-6; // µm: mask bottom offset from oxide top

  double bounds[2 * D] = {-cfg.xExtent, cfg.xExtent, cfg.yMin, cfg.yMax};
  BoundaryType bc[D] = {BoundaryType::REFLECTIVE_BOUNDARY,
                        BoundaryType::INFINITE_BOUNDARY};

  // Si substrate at y = 0.
  auto siLS = makePlane(bounds, bc, cfg.gridDelta, 0.);

  // Pad oxide: geometrically advance Si surface by padOxideThickness.
  auto oxLS = ls::Domain<NumericType, D>::New(siLS);
  auto sphere =
      ls::SmartPointer<ls::SphereDistribution<viennahrle::CoordType, D>>::New(
          cfg.padOxideThickness);
  ls::GeometricAdvect<NumericType, D>(oxLS, sphere).apply();

  // SiN mask: box occupying x < maskEdge, sitting flat on the pad oxide.
  // The tiny contact epsilon places the mask bottom numerically below the oxide
  // top so Cartesian stencils unambiguously hit the mask boundary.
  auto maskLS =
      makeMask(bounds, bc, cfg.gridDelta, -cfg.xExtent/2., cfg.xExtent/2.,
               cfg.padOxideThickness - maskContactEpsilon,
               cfg.padOxideThickness + cfg.maskThickness);

  // ── ViennaPS domain ───────────────────────────────────────────────────────

  auto domain = ps::Domain<NumericType, D>::New();
  domain->insertNextLevelSetAsMaterial(siLS, ps::Material::Si, false);
  domain->insertNextLevelSetAsMaterial(oxLS, ps::Material::SiO2, false);

  // Only add mask if thickness is positive. Setting maskThickness <= 0 disables
  // LOCOS physics and uses standard oxidation instead.
  if (cfg.maskThickness > NumericType(0)) {
    domain->insertNextLevelSetAsMaterial(maskLS, ps::Material::Si3N4, false);
  }

  // ── Oxidation model ───────────────────────────────────────────────────────

  auto model = ps::SmartPointer<ps::Oxidation<NumericType, D>>::New();
  model->setTemperature(cfg.temperature);
  model->setOxidant(cfg.oxidant == "wet" ? ps::OxidantType::Wet
                                         : ps::OxidantType::Dry);
  model->setPressure(cfg.pressure);
  model->setOrientation(cfg.orientation == "111"
                            ? ps::SiliconOrientation::Si111
                            : ps::SiliconOrientation::Si100);
  model->setTimeStep(cfg.timeStep);
  model->setMaxGridPoints(cfg.maxGridPoints);
  model->setMechanicsIterations(cfg.mechanicsIterations);
  model->setMechanicsTolerance(cfg.mechanicsTolerance);
  model->setPressureIterations(cfg.pressureIterations);
  model->setPressureTolerance(cfg.pressureTolerance);
  model->setStokesIterations(cfg.stokesIterations);
  model->setStokesTolerance(cfg.stokesTolerance);
  model->setCouplingIterations(cfg.couplingIterations);
  model->setCouplingTolerance(cfg.couplingTolerance);
  model->setMaskCouplingIterations(cfg.maskCouplingIterations);
  model->setMaskCouplingTolerance(cfg.maskCouplingTolerance);

  // LOCOS: mask material is already Si3N4 (default); just set parameters.
  model->setMaskParameters(
      viennals::OxidationPresets<NumericType>::siliconNitrideMask1000C());

  model->saveSurfaceMesh(domain, cfg.outputPrefix + "_stack_step_000.vtp");
  model->saveVolumeMesh(domain, cfg.outputPrefix + "_stack_step_000");

  const NumericType est =
      model->estimatePlanarOxideThickness(cfg.padOxideThickness);
  std::cout << "Planar Deal-Grove estimate for " << cfg.oxidationTime
            << " hr oxidation at " << cfg.temperature
            << " C: " << est << " um total oxide thickness.\n";

  NumericType elapsed = 0.;
  unsigned step = 0;
  const NumericType timeEps = 1e-9 * cfg.oxidationTime;
  while (cfg.oxidationTime - elapsed > timeEps) {
    NumericType dt = cfg.timeStep;
    if (elapsed + dt > cfg.oxidationTime)
      dt = cfg.oxidationTime - elapsed;
    if (dt <= NumericType(0))
      break;

    model->setTime(dt);
    model->setTimeStep(dt);
    ps::Process<NumericType, D>(domain, model, NumericType(0)).apply();

    elapsed += dt;
    ++step;

    std::ostringstream filename;
    filename << cfg.outputPrefix << "_stack_step_" << std::setw(3)
             << std::setfill('0') << step;// << ".vtp";
    model->saveSurfaceMesh(domain, filename.str() + ".vtp");
    model->saveVolumeMesh(domain, filename.str());
    std::cout << "Wrote " << filename.str() << " at t = " << elapsed
              << " hr.\n";
  }

  // ── Final output ──────────────────────────────────────────────────────────
  model->saveVolumeMesh(domain, cfg.outputPrefix + "_stack_after");

  std::cout << "Wrote " << cfg.outputPrefix << "_stack_initial.vtp, "
            << step << " time-step files, "
            << cfg.outputPrefix << "_stack_after.vtp, and "
            << cfg.outputPrefix << "_stack_after_volume.vtu\n";

  return 0;
}
