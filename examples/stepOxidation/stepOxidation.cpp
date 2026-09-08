#include <geometries/psMakeFin.hpp>
#include <models/psOxidation.hpp>
#include <process/psProcess.hpp>
#include <psDomain.hpp>
#include <psUtil.hpp>

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
// Simulation driver
//
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
  ps::Logger::setLogLevel(ps::LogLevel::ERROR);

  const NumericType gridDelta = params.get("gridDelta");
  const NumericType xExtent = params.get("xExtent");
  const NumericType finWidth = params.get("finWidth");
  const NumericType finHeight = params.get("finHeight");
  // yMin/yMax set the growth-direction bounds for the HRLE domain.  With an
  // INFINITE boundary the level set extends correctly regardless, so these are
  // only needed when the surface might reach the boundary during simulation.
  const NumericType yMin = params.get("yMin", NumericType(-2.0));
  const NumericType yMax = params.get("yMax", finHeight + NumericType(2.0));
  const NumericType oxideThickness =
      params.get("oxideThickness", NumericType(0));
  const NumericType oxidationTime = params.get("oxidationTime");
  const NumericType temperature = params.get("temperature");
  const NumericType pressure = params.get("pressure");

  const auto oxidant = params.get<std::string>("oxidant", "wet");
  const auto orientation = params.get<std::string>("orientation", "100");
  const auto outputPrefix =
      params.get<std::string>("outputPrefix", "ps_step_oxidation");

  double bounds[2 * D];
  ps::BoundaryType boundaryCons[D];

  bounds[0] = -xExtent;
  bounds[1] = xExtent;
  if constexpr (D == 2) {
    bounds[2] = yMin;
    bounds[3] = yMax;
    boundaryCons[0] = ps::BoundaryType::REFLECTIVE_BOUNDARY;
    boundaryCons[1] = ps::BoundaryType::INFINITE_BOUNDARY;
  } else {
    const NumericType zExtent = params.get("zExtent", xExtent);

    // Y = step extrusion (REFLECTIVE), Z = growth (INFINITE)
    bounds[2] = -zExtent;
    bounds[3] = zExtent;
    bounds[4] = yMin;
    bounds[5] = yMax;
    boundaryCons[0] = ps::BoundaryType::REFLECTIVE_BOUNDARY;
    boundaryCons[1] = ps::BoundaryType::REFLECTIVE_BOUNDARY;
    boundaryCons[2] = ps::BoundaryType::INFINITE_BOUNDARY;
  }

  // MakeFin with halfFin=true calls halveXAxis() on the setup, clipping the
  // domain to [0, xExtent].  The fin (raised platform) occupies x in
  // [0, finWidth/2] and the step wall sits at x = finWidth/2.
  auto domain =
      ps::Domain<NumericType, D>::New(bounds, boundaryCons, gridDelta);
  ps::MakeFin<NumericType, D>(domain, finWidth, finHeight,
                              /*taperAngle=*/NumericType(0),
                              /*maskHeight=*/NumericType(0),
                              /*maskTaperAngle=*/NumericType(0),
                              /*halfFin=*/true)
      .apply();

  // The deformation solver needs the oxide to be at least gridDelta thick so
  // that Cartesian solve nodes exist between the two surfaces.
  const NumericType seedThickness = std::max(oxideThickness, gridDelta);
  {
    auto ambientInterface =
        ls::Domain<NumericType, D>::New(domain->getLevelSets().back());
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
