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

#include <lsMakeGeometry.hpp>

#include <geometries/psMakePlane.hpp>
#include <models/psOxidation.hpp>
#include <process/psProcess.hpp>
#include <psDomain.hpp>

namespace ls = viennals;
namespace ps = viennaps;

using NumericType = double;
constexpr int D = 2;

int parseMaskContactMode(const std::string &value) {
  const auto mode = ps::util::detail::lower(value);
  if (mode == "0" || mode == "kinematic")
    return 0;
  if (mode == "1" || mode == "2" || mode == "oneway" || mode == "one-way" ||
      mode == "traction")
    return 1;
  if (mode == "3" || mode == "4" || mode == "elastic" || mode == "twoway" ||
      mode == "two-way" || mode == "two_way" || mode == "feedback" ||
      mode == "twoway-elastic" || mode == "two-way-elastic" ||
      mode == "elastic-feedback")
    return 2;

  std::cerr << "Warning: unknown maskContactMode='" << value
            << "', using oneway.\n";
  return 1;
}

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

  ps::Logger::setLogLevel(params.get<std::string>("logLevel", "info"));
  omp_set_num_threads(params.get<int>("numThreads"));

  // ── Geometry ──────────────────────────────────────────────────────────────

  // Asymmetric y-bounds: yMin below Si surface (for the substrate), yMax above
  // anticipated oxide height.  MakePlane reuses this setup for all layers.
  const NumericType gridDelta = params.get("gridDelta");
  const NumericType xExtent = params.get("xExtent");
  const NumericType yMin = params.get("yMin");
  const NumericType yMax = params.get("yMax");
  double bounds[2 * D] = {-xExtent, xExtent, yMin, yMax};
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
    auto maskLS = ls::Domain<NumericType, D>::New(bounds, bc, gridDelta);
    const ls::VectorType<NumericType, D> minCorner{-xExtent, padOxideThickness -
                                                                 contactEps};
    const ls::VectorType<NumericType, D> maxCorner{maskEdge, padOxideThickness +
                                                                 maskThickness};
    ls::MakeGeometry<NumericType, D> geom(
        maskLS, ls::Box<NumericType, D>::New(minCorner, maxCorner));
    geom.setIgnoreBoundaryConditions(std::array<bool, D>{false, true});
    geom.apply();
    domain->insertNextLevelSetAsMaterial(maskLS, ps::Material::Si3N4, false);
  }

  // ── Oxidation model ───────────────────────────────────────────────────────

  const NumericType temperature = params.get("temperature");
  const NumericType pressure = params.get("pressure");
  const NumericType oxidationTime = params.get("oxidationTime");
  const NumericType timeStep = params.get("timeStep");

  const std::string outputPrefix =
      params.get<std::string>("outputPrefix", "locosOxidation");

  auto model = ps::SmartPointer<ps::Oxidation<NumericType, D>>::New();
  model->setTemperature(temperature);
  model->setOxidant(params.get<std::string>("oxidant"));
  model->setPressure(pressure);
  model->setOrientation(params.get<std::string>("orientation"));
  model->setTimeStep(timeStep);
  if (params.contains("maxGridPoints"))
    model->setMaxGridPoints(params.get<unsigned>("maxGridPoints"));
  model->setMechanicsIterations(params.get<unsigned>("mechanicsIterations"));
  model->setMechanicsTolerance(params.get("mechanicsTolerance"));
  model->setPressureIterations(params.get<unsigned>("pressureIterations"));
  model->setPressureTolerance(params.get("pressureTolerance"));
  model->setStokesIterations(params.get<unsigned>("stokesIterations"));
  model->setStokesTolerance(params.get("stokesTolerance"));
  model->setCouplingIterations(params.get<unsigned>("couplingIterations"));
  model->setCouplingTolerance(params.get("couplingTolerance"));
  model->setMaskCouplingIterations(
      params.get<unsigned>("maskCouplingIterations"));
  model->setMaskCouplingTolerance(params.get("maskCouplingTolerance"));

  model->setGpuMode(params.get<std::string>("useGpu", "cpu"));
  model->setGpuPreconditioner(
      params.get<std::string>("gpuPreconditioner", "ilu0"));

  auto maskParams = viennals::OxidationPresets::siliconNitrideMask1000C();
  maskParams.referenceViscosity =
      std::max(params.get("maskReferenceViscosity"), 0.0);
  maskParams.youngModulus = std::max(params.get("maskYoungModulus"), 0.0);
  maskParams.poissonRatio = params.get("maskPoissonRatio");
  maskParams.contactMode =
      parseMaskContactMode(params.get<std::string>("maskContactMode"));
  maskParams.anchorBoundaryDirection =
      params.get<int>("maskAnchorBoundaryDirection");
  maskParams.anchorBoundarySide = params.get<int>("maskAnchorBoundarySide");
  maskParams.anchorBoundaryLayers = params.get<int>("maskAnchorBoundaryLayers");
  model->setMaskParameters(maskParams);
  model->setMaskTractionIterations(static_cast<unsigned>(
      std::max(1, params.get<int>("maskTractionIterations"))));
  model->setMaskTractionTolerance(params.get("maskTractionTolerance"));
  model->setMaskTractionRelaxation(params.get("maskTractionRelaxation"));
  model->setMaskContactLoadRelaxation(params.get("maskContactLoadRelaxation"));
  model->setMaskContactReleaseFraction(
      params.get("maskContactReleaseFraction"));
  model->setMaskUnilateralContact(params.get<bool>("maskUnilateralContact"));
  model->setMaskSmootherOmega(params.get("maskSmootherOmega"));

  model->saveSurfaceMesh(domain, outputPrefix + "_stack_step_000.vtp");
  model->saveVolumeMesh(domain, outputPrefix + "_stack_step_000");

  const NumericType est =
      model->estimatePlanarOxideThickness(padOxideThickness);
  std::cout << "Planar Deal-Grove estimate for " << oxidationTime
            << " hr oxidation at " << temperature << " C: " << est
            << " um total oxide thickness.\n";

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
    model->saveSurfaceMesh(domain, filename.str() + ".vtp");
    model->saveVolumeMesh(domain, filename.str());
    std::cout << "Wrote " << filename.str() << " at t = " << elapsed
              << " hr.\n";
  }

  // ── Final output ──────────────────────────────────────────────────────────
  model->saveVolumeMesh(domain, outputPrefix + "_stack_after");

  std::cout << "Wrote " << outputPrefix << "_stack_initial.vtp, " << step
            << " time-step files, " << outputPrefix << "_stack_after.vtp, and "
            << outputPrefix << "_stack_after_volume.vtu\n";

  return 0;
}
