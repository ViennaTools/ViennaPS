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

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
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
  NumericType maskReferenceViscosity = -1.; // Pa·hr; <0 uses preset
  NumericType maskPoissonRatio = 0.27;
  // Mask contact mode: "traction" (default) or "kinematic" (legacy).
  std::string maskContactMode = "traction";
  bool maskUnilateralContact = true;
  int maskTractionIterations = 10000;
  NumericType maskTractionTolerance = 1e-5;
  NumericType maskTractionRelaxation = 0.9;
  NumericType maskContactLoadRelaxation = 0.25;
  NumericType maskContactReleaseFraction = 5e-3;
  NumericType maskSmootherOmega = 1.0;   // SOR omega for multigrid V-cycle smoother
  int maskAnchorBoundaryDirection = 0; // x direction in this 2D LOCOS setup
  int maskAnchorBoundarySide = -1;     // -1: far-left mask edge; 0 disables
  unsigned maskAnchorBoundaryLayers = 1;
  // "auto" (default, GPU when n>=threshold), "gpu" (always GPU), "cpu" (always CPU)
  std::string useGpu = "auto";
  // "jacobi" matches the CPU diffusion preconditioner; "ilu0" is experimental.
  std::string gpuPreconditioner = "jacobi";
  // "debug", "timing", "intermediate", "info" (default), "warning", "error"
  std::string logLevel = "info";
};

bool parseBool(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value == "1" || value == "true" || value == "yes" || value == "on";
}

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
    else if (key == "maskReferenceViscosity") cfg.maskReferenceViscosity = std::stod(val);
    else if (key == "maskPoissonRatio") cfg.maskPoissonRatio = std::stod(val);
    else if (key == "maskContactMode") cfg.maskContactMode = val;
    else if (key == "maskUnilateralContact") cfg.maskUnilateralContact = parseBool(val);
    else if (key == "maskTractionIterations") cfg.maskTractionIterations = std::stoi(val);
    else if (key == "maskTractionTolerance") cfg.maskTractionTolerance = std::stod(val);
    else if (key == "maskTractionRelaxation") cfg.maskTractionRelaxation = std::stod(val);
    else if (key == "maskContactLoadRelaxation") cfg.maskContactLoadRelaxation = std::stod(val);
    else if (key == "maskContactReleaseFraction") cfg.maskContactReleaseFraction = std::stod(val);
    else if (key == "maskSmootherOmega")      cfg.maskSmootherOmega      = std::stod(val);
    else if (key == "maskAnchorBoundaryDirection") cfg.maskAnchorBoundaryDirection = std::stoi(val);
    else if (key == "maskAnchorBoundarySide") cfg.maskAnchorBoundarySide = std::stoi(val);
    else if (key == "maskAnchorBoundaryLayers") cfg.maskAnchorBoundaryLayers = std::stoul(val);
    else if (key == "useGpu")                cfg.useGpu = val;
    else if (key == "gpuPreconditioner")     cfg.gpuPreconditioner = val;
    else if (key == "logLevel")              cfg.logLevel = val;
  }
  return cfg;
}

// ── Main ─────────────────────────────────────────────────────────────────────

int main() {
  using Clock = std::chrono::steady_clock;
  const auto wallStart = Clock::now();
  std::chrono::duration<double> meshWriteTime{0.};
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
      makeMask(bounds, bc, cfg.gridDelta, -cfg.xExtent, cfg.maskEdge,
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

  if      (cfg.useGpu == "gpu") model->setGpuMode(ps::GpuMode::Gpu);
  else if (cfg.useGpu == "cpu") model->setGpuMode(ps::GpuMode::Cpu);
  // else "auto": leave the default (GpuMode::Auto = GPU when n >= threshold)
  if      (cfg.gpuPreconditioner == "ilu0")
    model->setGpuPreconditioner(ps::GpuPreconditioner::ILU0);
  else
    model->setGpuPreconditioner(ps::GpuPreconditioner::Jacobi);

  // LOCOS: mask material is already Si3N4 (default); just set parameters.
  auto maskParams =
      viennals::OxidationPresets<NumericType>::siliconNitrideMask1000C();
  if (cfg.maskReferenceViscosity > NumericType(0))
    maskParams.referenceViscosity = cfg.maskReferenceViscosity;
  maskParams.poissonRatio = cfg.maskPoissonRatio;
  maskParams.contactMode =
      (cfg.maskContactMode == "kinematic") ? 0 :
      (cfg.maskContactMode == "oneway")    ? 1 : 2;
  maskParams.anchorBoundaryDirection = cfg.maskAnchorBoundaryDirection;
  maskParams.anchorBoundarySide = cfg.maskAnchorBoundarySide;
  maskParams.anchorBoundaryLayers = cfg.maskAnchorBoundaryLayers;
  model->setMaskParameters(maskParams);
  model->setMaskTractionIterations(
      static_cast<unsigned>(std::max(1, cfg.maskTractionIterations)));
  model->setMaskTractionTolerance(cfg.maskTractionTolerance);
  model->setMaskTractionRelaxation(cfg.maskTractionRelaxation);
  model->setMaskContactLoadRelaxation(cfg.maskContactLoadRelaxation);
  model->setMaskContactReleaseFraction(cfg.maskContactReleaseFraction);
  model->setMaskUnilateralContact(cfg.maskUnilateralContact);
  model->setMaskSmootherOmega(cfg.maskSmootherOmega);

  {
    const auto meshWriteStart = Clock::now();
    model->saveSurfaceMesh(domain, cfg.outputPrefix + "_stack_step_000.vtp");
    model->saveVolumeMesh(domain, cfg.outputPrefix + "_stack_step_000");
    meshWriteTime += Clock::now() - meshWriteStart;
  }

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
    {
      const auto meshWriteStart = Clock::now();
      model->saveSurfaceMesh(domain, filename.str() + ".vtp");
      model->saveVolumeMesh(domain, filename.str());
      meshWriteTime += Clock::now() - meshWriteStart;
    }
    std::cout << "Wrote " << filename.str() << " at t = " << elapsed
              << " hr.\n";
  }

  // ── Final output ──────────────────────────────────────────────────────────
  {
    const auto meshWriteStart = Clock::now();
    model->saveVolumeMesh(domain, cfg.outputPrefix + "_stack_after");
    meshWriteTime += Clock::now() - meshWriteStart;
  }

  std::cout << "Wrote " << cfg.outputPrefix << "_stack_initial.vtp, "
            << step << " time-step files, "
            << cfg.outputPrefix << "_stack_after.vtp, and "
            << cfg.outputPrefix << "_stack_after_volume.vtu\n";

  const auto wallEnd = Clock::now();
  const std::chrono::duration<double> wallTime =
      (wallEnd - wallStart) - meshWriteTime;
  std::cout << "Total LOCOS wall time excluding mesh writes: "
            << std::fixed << std::setprecision(3)
            << wallTime.count() << " s\n";

  return 0;
}
