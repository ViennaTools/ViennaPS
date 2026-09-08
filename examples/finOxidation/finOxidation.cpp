#include <geometries/psMakeFin.hpp>
#include <models/psGeometricDistributionModels.hpp>
#include <models/psOxidation.hpp>
#include <process/psProcess.hpp>
#include <psDomain.hpp>
#include <psUtil.hpp>

namespace ps = viennaps;

using NumericType = double;

// ---------------------------------------------------------------------------
// Coordinate convention:
//   2D: X = lateral (REFLECTIVE), Y = growth (INFINITE)
//   3D: X = lateral (REFLECTIVE), Y = fin extrusion (REFLECTIVE), Z = growth
//   (INFINITE)
// ---------------------------------------------------------------------------

template <int D> void run(const ps::util::Parameters &params) {
  omp_set_num_threads(params.get<int>("numThreads"));
  ps::Logger::setLogLevel(ps::LogLevel::ERROR);

  const NumericType gridDelta = params.get("gridDelta");
  const NumericType xExtent = 2 * params.get("xExtent");
  const NumericType yExtent = 2 * params.get("zExtent", xExtent / 2);
  const NumericType finWidth = params.get("finWidth");
  const NumericType finHeight = params.get("finHeight");
  const NumericType oxideThickness =
      params.get("oxideThickness", NumericType(0.0));
  const NumericType oxidationTime = params.get("oxidationTime");
  const NumericType temperature = params.get("temperature");
  const NumericType pressure = params.get("pressure");

  const auto oxidant = params.get<std::string>("oxidant", "wet");
  const auto orientation = params.get<std::string>("orientation", "100");
  const auto outputPrefix =
      params.get<std::string>("outputPrefix", "ps_fin_oxidation");

  auto domain = ps::Domain<NumericType, D>::New(
      gridDelta, xExtent, yExtent, ps::BoundaryType::REFLECTIVE_BOUNDARY);
  ps::MakeFin<NumericType, D>(domain, finWidth, finHeight).apply();

  // Clamp the oxide seed to at least gridDelta so the Cartesian solve always
  // has resolvable nodes between the Si and SiO2 level sets.
  const NumericType seedThickness = std::max(oxideThickness, gridDelta);
  domain->duplicateTopLevelSet(ps::Material::SiO2);
  ps::Process<NumericType, D>(
      domain, ps::SmartPointer<ps::SphereDistribution<NumericType, D>>::New(
                  seedThickness))
      .apply();

  auto model = ps::SmartPointer<ps::Oxidation<NumericType, D>>::New();
  model->setTemperature(temperature);
  model->setTime(oxidationTime);
  model->setOxidant(oxidant);
  model->setPressure(pressure);
  model->setOrientation(orientation);
  model->setInitialOxideThickness(seedThickness);

  model->setGpuMode(params.get<std::string>("useGpu", "cpu"));
  model->setGpuPreconditioner(
      params.get<std::string>("gpuPreconditioner", "jacobi"));

  if (params.contains("maxGridPoints"))
    model->setMaxGridPoints(params.get<unsigned>("maxGridPoints"));

  model->saveSurfaceMesh(domain, outputPrefix + "_stack_initial.vtp");
  model->saveVolumeMesh(domain, outputPrefix + "_stack_initial");

  const auto t0 = std::chrono::steady_clock::now();
  ps::Process<NumericType, D>(domain, model, NumericType(0)).apply();
  const double elapsedSim =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();

  model->saveSurfaceMesh(domain, outputPrefix + "_stack_after.vtp");
  model->saveVolumeMesh(domain, outputPrefix + "_stack_after");

  std::cout << "Simulation time: " << elapsedSim << " s\n";
  std::cout << "Planar Deal-Grove estimate for " << oxidationTime
            << " hr oxidation at " << temperature
            << " C: " << model->estimatePlanarOxideThickness(seedThickness)
            << " um oxide thickness." << std::endl;

  std::cout << "Wrote " << outputPrefix << "_stack_initial.vtp and "
            << outputPrefix << "_stack_after.vtp and " << outputPrefix
            << "_stack_after_volume.vtu" << std::endl;
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
