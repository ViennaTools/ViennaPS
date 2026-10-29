#include <geometries/psMakeFin.hpp>
#include <models/psOxidation.hpp>
#include <process/psProcess.hpp>
#include <psDomain.hpp>
#include <psUtil.hpp>
#include <vcTimer.hpp>

#include <sstream>

namespace ps = viennaps;

using NumericType = double;

// ---------------------------------------------------------------------------
// Geometry: a rectangular Si fin (half-fin) centred at the reflective boundary
// x = 0, with the step wall at x = finWidth / 2.  In the visible simulation
// domain [0, xExtent] the raised platform occupies [0, finWidth/2] and the
// flat substrate occupies [finWidth/2, xExtent].  The reflective boundary
// mirrors the fin symmetrically; oxide grows on the top, both sides of the
// fin wall, and the surrounding substrate.
//
// Coordinate convention:
//   2D: X = lateral (REFLECTIVE at x=0), Y = growth (INFINITE)
//   3D: X = lateral (REFLECTIVE at x=0), Y = step extrusion (REFLECTIVE),
//       Z = growth (INFINITE)
// ---------------------------------------------------------------------------

template <int D> void run(const ps::util::Parameters &params) {
  omp_set_num_threads(params.get<int>("numThreads"));
  if (params.contains("logLevel"))
    ps::Logger::setLogLevel(params.get<std::string>("logLevel"));

  const NumericType gridDelta = params.get("gridDelta");
  const NumericType xExtent = 2 * params.get("xExtent");
  const NumericType yExtent = 2 * params.get("zExtent", xExtent / 2);
  const NumericType finWidth = params.get("finWidth");
  const NumericType finHeight = params.get("finHeight");
  const NumericType oxideThickness =
      params.get("oxideThickness", NumericType(0));
  const NumericType oxidationTime = params.get("oxidationTime");
  const NumericType temperature = params.get("temperature");
  const NumericType pressure = params.get("pressure");

  const auto oxidant = params.get<std::string>("oxidant", "wet");
  const auto orientation = params.get<std::string>("orientation", "100");
  const auto outputPrefix =
      params.get<std::string>("outputPrefix", "ps_step_oxidation");

  // MakeFin with halfFin=true calls halveXAxis() on the setup, clipping the
  // domain to [0, xExtent].  The fin (raised platform) occupies x in
  // [0, finWidth/2] and the step wall sits at x = finWidth/2.
  auto domain = ps::Domain<NumericType, D>::New(
      gridDelta, xExtent, yExtent, ps::BoundaryType::REFLECTIVE_BOUNDARY);
  ps::MakeFin<NumericType, D>(domain, finWidth, finHeight,
                              /*taperAngle=*/NumericType(0),
                              /*maskHeight=*/NumericType(0),
                              /*maskTaperAngle=*/NumericType(0),
                              /*halfFin=*/true)
      .apply();

  auto model = ps::SmartPointer<ps::Oxidation<NumericType, D>>::New();
  model->setTemperature(temperature);
  model->setTime(oxidationTime);
  model->setOxidant(oxidant);
  model->setPressure(pressure);
  model->setOrientation(orientation);
  // The oxide on the silicon at t = 0 (0 = bare silicon).  The model creates
  // it and, while it is thinner than three grid cells, grows it by the 1-D
  // Deal-Grove law before the 2-D solve starts.
  model->setInitialOxideThickness(oxideThickness);

  model->setGpuMode(params.get<std::string>("useGpu", "cpu"));

  if (params.contains("maxGridPoints"))
    model->setMaxGridPoints(params.get<unsigned>("maxGridPoints"));

  domain->saveSurfaceMesh(outputPrefix + "_stack_initial.vtp");
  domain->saveVolumeMesh(outputPrefix + "_stack_initial");

  ps::Timer<> timer;
  timer.start();
  ps::Process<NumericType, D>(domain, model, NumericType(0)).apply();
  timer.finish();

  model->saveSurfaceMesh(domain, outputPrefix + "_stack_after.vtp");
  model->saveVolumeMesh(domain, outputPrefix + "_stack_after");

  std::ostringstream estimate;
  estimate << "Planar Deal-Grove estimate for " << oxidationTime
           << " hr oxidation at " << temperature
           << " C: " << model->estimatePlanarOxideThickness(oxideThickness)
           << " um oxide thickness.";
  ps::Logger::getInstance()
      .addInfo("Simulation time: " +
               std::to_string(1e-9 * timer.currentDuration) + " s")
      .addInfo(estimate.str())
      .addInfo("Wrote " + outputPrefix + "_stack_initial.vtp, " + outputPrefix +
               "_stack_after.vtp and " + outputPrefix +
               "_stack_after_volume.vtu")
      .print();
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

  const int dimensions = params.get<int>("dimensions", 2);
  if (dimensions == 3)
    run<3>(params);
  else
    run<2>(params);

  return 0;
}
