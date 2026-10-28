// LOCOS (Local Oxidation of Silicon) process example using psOxidation.
//
// Geometry:
//   - Si substrate (plane at y = 0)
//   - Pad oxide (SiO2, thickness padOxideThickness) grown on Si before masking
//   - Si3N4 mask box (covering x < maskEdge, sitting on top of pad oxide)
//
// The Oxidation model auto-detects the Si3N4 material and activates LOCOS
// physics: the nitride joins the oxide mechanics solve as an elastic body
// that creeps, producing the characteristic bird's-beak oxide profile.
//
// Config keys (lengths in µm, time in hours, pressure in atm): see
// knownKeys below and config.txt.
//
// `timeStep` controls the output cadence and the maximum oxidation substep.
// The model automatically uses smaller CFL-limited physics steps if needed.

#include <lsMakeGeometry.hpp>

#include <geometries/psMakePlane.hpp>
#include <models/psOxidation.hpp>
#include <process/psProcess.hpp>
#include <psDomain.hpp>
#include <vcTimer.hpp>

#include <iomanip>
#include <set>
#include <sstream>

namespace ls = viennals;
namespace ps = viennaps;

using NumericType = double;
constexpr int D = 2;

int main(int argc, char **argv) {
  ps::Timer<> wallTimer, meshTimer;
  wallTimer.start();

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

  if (params.contains("logLevel"))
    ps::Logger::setLogLevel(params.get<std::string>("logLevel"));
  auto &log = ps::Logger::getInstance();
  omp_set_num_threads(params.get<int>("numThreads"));

  // Keys this example reads; anything else is a typo or a retired key.
  const std::set<std::string> knownKeys = {
      "numThreads",          "logLevel",
      "useGpu",              "gridDelta",
      "xExtent",             "yMin",
      "yMax",                "padOxideThickness",
      "maskThickness",       "maskEdge",
      "oxidationTime",       "timeStep",
      "temperature",         "pressure",
      "oxidant",             "orientation",
      "maxGridPoints",       "outputPrefix",
      "mechanicsIterations", "mechanicsTolerance",
      "couplingIterations",  "couplingTolerance",
      "maskYoungModulus",    "maskPoissonRatio",
      "maskViscosity",       "maskShearActivationVolume"};
  for (const auto &entry : params.m)
    if (knownKeys.count(entry.first) == 0)
      log.addWarning("Unknown config key '" + entry.first + "' ignored.")
          .print();

  // ── Geometry ──────────────────────────────────────────────────────────────

  // Asymmetric y-bounds: yMin below Si surface (for the substrate), yMax above
  // anticipated oxide height.  MakePlane reuses this setup for all layers.
  const NumericType gridDelta = params.get("gridDelta");
  const NumericType xExtent = params.get("xExtent");
  double bounds[2 * D] = {-xExtent, xExtent, params.get("yMin"),
                          params.get("yMax")};
  ps::BoundaryType bc[D] = {ps::BoundaryType::REFLECTIVE_BOUNDARY,
                            ps::BoundaryType::INFINITE_BOUNDARY};

  auto domain = ps::Domain<NumericType, D>::New(bounds, bc, gridDelta);

  // Si substrate flat at y = 0.
  ps::MakePlane<NumericType, D>(domain, 0., ps::Material::Si).apply();

  // Pad SiO2: flat plane at y = padOxideThickness grown on the Si surface.
  const NumericType padOxideThickness = params.get("padOxideThickness");
  ps::MakePlane<NumericType, D>(domain, padOxideThickness, ps::Material::SiO2,
                                /*addToExisting=*/true)
      .apply();

  // Si3N4 mask: box covering x ∈ [−xExtent, maskEdge], sitting on the pad
  // oxide.  The tiny contact epsilon ensures the mask bottom is numerically
  // inside the oxide so Cartesian stencils unambiguously hit the boundary.
  const NumericType maskThickness = params.get("maskThickness");
  const NumericType maskEdge = params.get("maskEdge");
  if (maskThickness > NumericType(0)) {
    constexpr NumericType contactEps = 1e-6; // µm
    auto maskLS = ls::Domain<NumericType, D>::New(domain->getGrid());
    // Overhang the left domain edge: a box ending exactly on the reflective
    // boundary gives the mask a face on the symmetry plane and leaves
    // uncovered oxide cells there.
    const ls::VectorType<NumericType, D> minCorner{
        -2 * xExtent, padOxideThickness - contactEps};
    const ls::VectorType<NumericType, D> maxCorner{
        maskEdge, padOxideThickness + maskThickness};
    ls::MakeGeometry<NumericType, D> geom(
        maskLS, ls::Box<NumericType, D>::New(minCorner, maxCorner));
    geom.setIgnoreBoundaryConditions(std::array<bool, D>{false, true});
    geom.apply();
    domain->insertNextLevelSetAsMaterial(maskLS, ps::Material::Si3N4, false);
  }

  // ── Oxidation model ───────────────────────────────────────────────────────

  const NumericType temperature = params.get("temperature");
  const NumericType oxidationTime = params.get("oxidationTime");
  const NumericType timeStep = params.get("timeStep");
  const auto outputPrefix = params.get<std::string>("outputPrefix");

  auto model = ps::SmartPointer<ps::Oxidation<NumericType, D>>::New();
  model->setTemperature(temperature);
  model->setOxidant(params.get<std::string>("oxidant"));
  model->setPressure(params.get("pressure"));
  model->setOrientation(params.get<std::string>("orientation"));
  model->setTimeStep(timeStep);
  if (params.contains("maxGridPoints"))
    model->setMaxGridPoints(params.get<std::size_t>("maxGridPoints"));
  model->setMechanicsIterations(params.get<unsigned>("mechanicsIterations"));
  model->setMechanicsTolerance(params.get("mechanicsTolerance"));
  model->setCouplingIterations(params.get<unsigned>("couplingIterations"));
  model->setCouplingTolerance(params.get("couplingTolerance"));
  model->setGpuMode(params.get<std::string>("useGpu", "cpu"));

  // Nitride: elastic with Maxwell creep.  maskYoungModulus <= 0 and
  // maskViscosity / maskShearActivationVolume < 0 take the Si3N4 preset at
  // the process temperature (Senez 1996); 0 = linear-elastic mask / linear
  // viscosity.
  auto maskParams = ls::OxidationPresets::siliconNitrideMask(temperature);
  if (params.get("maskYoungModulus") > NumericType(0))
    maskParams.youngModulus = params.get("maskYoungModulus");
  maskParams.poissonRatio = params.get("maskPoissonRatio");
  if (params.get("maskViscosity") >= NumericType(0))
    maskParams.viscosity = params.get("maskViscosity");
  if (params.get("maskShearActivationVolume") >= NumericType(0))
    maskParams.shearActivationVolume = params.get("maskShearActivationVolume");
  model->setMaskParameters(maskParams);

  meshTimer.start();
  model->saveSurfaceMesh(domain, outputPrefix + "_stack_step_000.vtp");
  model->saveVolumeMesh(domain, outputPrefix + "_stack_step_000");
  meshTimer.finish();

  model->setTime(oxidationTime);
  const NumericType est =
      model->estimatePlanarOxideThickness(padOxideThickness);
  std::ostringstream message;
  message << "Planar Deal-Grove estimate for " << oxidationTime
          << " hr oxidation at " << temperature << " C: " << est
          << " um total oxide thickness.";
  log.addInfo(message.str()).print();

  // ── Time-stepping loop ────────────────────────────────────────────────────

  NumericType elapsed = 0.;
  unsigned step = 0;
  const NumericType timeEps = 1e-9 * oxidationTime;
  while (oxidationTime - elapsed > timeEps) {
    NumericType dt = timeStep;
    if (elapsed + dt > oxidationTime)
      dt = oxidationTime - elapsed;
    if (dt <= NumericType(0))
      break;

    model->setTime(dt);
    model->setTimeStep(dt);
    ps::Process<NumericType, D>(domain, model, NumericType(0)).apply();

    elapsed += dt;
    ++step;

    std::ostringstream filename;
    filename << outputPrefix << "_stack_step_" << std::setw(3)
             << std::setfill('0') << step;
    meshTimer.start();
    model->saveSurfaceMesh(domain, filename.str() + ".vtp");
    model->saveVolumeMesh(domain, filename.str());
    meshTimer.finish();
    log.addInfo("Wrote " + filename.str() + " at t = " +
                std::to_string(elapsed) + " hr.")
        .print();
  }

  // ── Final output ──────────────────────────────────────────────────────────
  meshTimer.start();
  model->saveVolumeMesh(domain, outputPrefix + "_stack_after");
  meshTimer.finish();
  log.addInfo("Wrote " + outputPrefix + "_stack_step_000.vtp, " +
              std::to_string(step) + " time-step files and " + outputPrefix +
              "_stack_after_volume.vtu")
      .print();

  wallTimer.finish();
  message.str("");
  message << "Total LOCOS wall time excluding mesh writes: " << std::fixed
          << std::setprecision(3)
          << 1e-9 * static_cast<double>(wallTimer.totalDuration -
                                        meshTimer.totalDuration)
          << " s";
  log.addInfo(message.str()).print();

  return 0;
}
