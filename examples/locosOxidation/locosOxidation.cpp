// LOCOS (Local Oxidation of Silicon) process example using psOxidation.
//
// Geometry:
//   - Si substrate (plane at y = 0)
//   - Pad oxide (SiO2, thickness padOxideThickness) grown on Si before masking
//   - Si3N4 mask box (covering x < maskEdge, sitting on top of pad oxide)
//
// The Oxidation model auto-detects the Si3N4 material and activates LOCOS
// physics: the nitride joins the oxide mechanics solve as a linear-elastic
// body, producing the characteristic bird's-beak oxide profile.
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
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <sstream>
#include <string>

#include <lsMakeGeometry.hpp>

#include <geometries/psMakePlane.hpp>
#include <models/psOxidation.hpp>
#include <process/psProcess.hpp>
#include <psDomain.hpp>

namespace ls = viennals;
namespace ps = viennaps;

using NumericType = double;
constexpr int D = 2;

using BoundaryType = ps::BoundaryType;

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
  int mechanicsIterations = 300;
  NumericType mechanicsTolerance = 1e-4;
  int couplingIterations = 8;
  NumericType couplingTolerance = 1e-6;
  NumericType maskYoungModulus = -1.; // Pa; <0 uses preset (270e9)
  NumericType maskPoissonRatio = 0.27;
  // Pa hr and m^3; <0 uses the Si3N4 preset at the process temperature
  // (Senez 1996), 0 = linear-elastic mask / linear viscosity
  NumericType maskViscosity = -1.;
  NumericType maskShearActivationVolume = -1.;
  std::string useGpu = "cpu";
  std::string gpuPreconditioner = "jacobi";
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
    if (key == "numThreads")
      cfg.numThreads = std::stoi(val);
    else if (key == "gridDelta")
      cfg.gridDelta = std::stod(val);
    else if (key == "xExtent")
      cfg.xExtent = std::stod(val);
    else if (key == "yMin")
      cfg.yMin = std::stod(val);
    else if (key == "yMax")
      cfg.yMax = std::stod(val);
    else if (key == "padOxideThickness")
      cfg.padOxideThickness = std::stod(val);
    else if (key == "maskThickness")
      cfg.maskThickness = std::stod(val);
    else if (key == "maskEdge")
      cfg.maskEdge = std::stod(val);
    else if (key == "oxidationTime")
      cfg.oxidationTime = std::stod(val);
    else if (key == "timeStep")
      cfg.timeStep = std::stod(val);
    else if (key == "temperature")
      cfg.temperature = std::stod(val);
    else if (key == "pressure")
      cfg.pressure = std::stod(val);
    else if (key == "oxidant")
      cfg.oxidant = val;
    else if (key == "orientation")
      cfg.orientation = val;
    else if (key == "maxGridPoints")
      cfg.maxGridPoints = std::stoull(val);
    else if (key == "outputPrefix")
      cfg.outputPrefix = val;
    else if (key == "mechanicsIterations")
      cfg.mechanicsIterations = std::stoi(val);
    else if (key == "mechanicsTolerance")
      cfg.mechanicsTolerance = std::stod(val);
    else if (key == "couplingIterations")
      cfg.couplingIterations = std::stoi(val);
    else if (key == "couplingTolerance")
      cfg.couplingTolerance = std::stod(val);
    else if (key == "maskYoungModulus")
      cfg.maskYoungModulus = std::stod(val);
    else if (key == "maskPoissonRatio")
      cfg.maskPoissonRatio = std::stod(val);
    else if (key == "maskViscosity")
      cfg.maskViscosity = std::stod(val);
    else if (key == "maskShearActivationVolume")
      cfg.maskShearActivationVolume = std::stod(val);
    else if (key == "useGpu")
      cfg.useGpu = val;
    else if (key == "gpuPreconditioner")
      cfg.gpuPreconditioner = val;
    else if (key == "logLevel")
      cfg.logLevel = val;
    else
      std::cerr << "Warning: unknown config key '" << key << "' ignored.\n";
  }
  return cfg;
}

// ── Main ─────────────────────────────────────────────────────────────────────

int main() {
  using Clock = std::chrono::steady_clock;
  const auto wallStart = Clock::now();
  std::chrono::duration<double> meshWriteTime{0.};
  const auto cfg = parseConfig("config.txt");

  {
    const auto &lv = cfg.logLevel;
    if (lv == "debug")
      ps::Logger::setLogLevel(ps::LogLevel::DEBUG);
    else if (lv == "timing")
      ps::Logger::setLogLevel(ps::LogLevel::TIMING);
    else if (lv == "intermediate")
      ps::Logger::setLogLevel(ps::LogLevel::INTERMEDIATE);
    else if (lv == "info")
      ps::Logger::setLogLevel(ps::LogLevel::INFO);
    else if (lv == "warning")
      ps::Logger::setLogLevel(ps::LogLevel::WARNING);
    else if (lv == "error")
      ps::Logger::setLogLevel(ps::LogLevel::ERROR);
    else
      ps::Logger::setLogLevel(ps::LogLevel::INFO);
  }
  omp_set_num_threads(cfg.numThreads);

  // ── Geometry ──────────────────────────────────────────────────────────────

  // Asymmetric y-bounds: yMin below Si surface (for the substrate), yMax above
  // anticipated oxide height.  MakePlane reuses this setup for all layers.
  double bounds[2 * D] = {-cfg.xExtent, cfg.xExtent, cfg.yMin, cfg.yMax};
  BoundaryType bc[D] = {BoundaryType::REFLECTIVE_BOUNDARY,
                        BoundaryType::INFINITE_BOUNDARY};

  auto domain = ps::Domain<NumericType, D>::New(bounds, bc, cfg.gridDelta);

  // Si substrate flat at y = 0.
  ps::MakePlane<NumericType, D>(domain, 0., ps::Material::Si).apply();

  // Pad SiO2: flat plane at y = padOxideThickness grown on the Si surface.
  ps::MakePlane<NumericType, D>(domain, cfg.padOxideThickness,
                                ps::Material::SiO2, /*addToExisting=*/true)
      .apply();

  // Si3N4 mask: box covering x ∈ [−xExtent, maskEdge], sitting on the pad
  // oxide.  The tiny contact epsilon ensures the mask bottom is numerically
  // inside the oxide so Cartesian stencils unambiguously hit the boundary.
  if (cfg.maskThickness > NumericType(0)) {
    constexpr NumericType contactEps = 1e-6; // µm
    auto maskLS = ls::Domain<NumericType, D>::New(bounds, bc, cfg.gridDelta);
    // Overhang the left domain edge: a box ending exactly on the reflective
    // boundary gives the mask a face on the symmetry plane and leaves
    // uncovered oxide cells there.
    const ls::VectorType<NumericType, D> minCorner{
        -2 * cfg.xExtent, cfg.padOxideThickness - contactEps};
    const ls::VectorType<NumericType, D> maxCorner{
        cfg.maskEdge, cfg.padOxideThickness + cfg.maskThickness};
    ls::MakeGeometry<NumericType, D> geom(
        maskLS, ls::Box<NumericType, D>::New(minCorner, maxCorner));
    geom.setIgnoreBoundaryConditions(std::array<bool, D>{false, true});
    geom.apply();
    domain->insertNextLevelSetAsMaterial(maskLS, ps::Material::Si3N4, false);
  }

  // ── Oxidation model ───────────────────────────────────────────────────────

  auto model = ps::SmartPointer<ps::Oxidation<NumericType, D>>::New();
  model->setTemperature(cfg.temperature);
  model->setOxidant(cfg.oxidant == "wet" ? ps::OxidantType::Wet
                                         : ps::OxidantType::Dry);
  model->setPressure(cfg.pressure);
  model->setOrientation(cfg.orientation == "111" ? ps::SiliconOrientation::Si111
                        : cfg.orientation == "110"
                            ? ps::SiliconOrientation::Si110
                            : ps::SiliconOrientation::Si100);
  model->setTimeStep(cfg.timeStep);
  model->setMaxGridPoints(cfg.maxGridPoints);
  model->setMechanicsIterations(cfg.mechanicsIterations);
  model->setMechanicsTolerance(cfg.mechanicsTolerance);
  model->setCouplingIterations(cfg.couplingIterations);
  model->setCouplingTolerance(cfg.couplingTolerance);

  if (cfg.useGpu == "gpu")
    model->setGpuMode(ps::GpuMode::Gpu);
  else if (cfg.useGpu == "cpu")
    model->setGpuMode(ps::GpuMode::Cpu);
  if (cfg.gpuPreconditioner == "ilu0")
    model->setGpuPreconditioner(ps::GpuPreconditioner::ILU0);
  else
    model->setGpuPreconditioner(ps::GpuPreconditioner::Jacobi);

  auto maskParams =
      viennals::OxidationPresets::siliconNitrideMask(cfg.temperature);
  if (cfg.maskYoungModulus > NumericType(0))
    maskParams.youngModulus = cfg.maskYoungModulus;
  maskParams.poissonRatio = cfg.maskPoissonRatio;
  if (cfg.maskViscosity >= NumericType(0))
    maskParams.viscosity = cfg.maskViscosity;
  if (cfg.maskShearActivationVolume >= NumericType(0))
    maskParams.shearActivationVolume = cfg.maskShearActivationVolume;
  model->setMaskParameters(maskParams);

  {
    const auto meshWriteStart = Clock::now();
    model->saveSurfaceMesh(domain, cfg.outputPrefix + "_stack_step_000.vtp");
    model->saveVolumeMesh(domain, cfg.outputPrefix + "_stack_step_000");
    meshWriteTime += Clock::now() - meshWriteStart;
  }

  model->setTime(cfg.oxidationTime);
  const NumericType est =
      model->estimatePlanarOxideThickness(cfg.padOxideThickness);
  std::cout << "Planar Deal-Grove estimate for " << cfg.oxidationTime
            << " hr oxidation at " << cfg.temperature << " C: " << est
            << " um total oxide thickness.\n";

  // ── Time-stepping loop ────────────────────────────────────────────────────

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
             << std::setfill('0') << step;
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

  std::cout << "Wrote " << cfg.outputPrefix << "_stack_initial.vtp, " << step
            << " time-step files, " << cfg.outputPrefix
            << "_stack_after.vtp, and " << cfg.outputPrefix
            << "_stack_after_volume.vtu\n";

  const auto wallEnd = Clock::now();
  const std::chrono::duration<double> wallTime =
      (wallEnd - wallStart) - meshWriteTime;
  std::cout << "Total LOCOS wall time excluding mesh writes: " << std::fixed
            << std::setprecision(3) << wallTime.count() << " s\n";

  return 0;
}
