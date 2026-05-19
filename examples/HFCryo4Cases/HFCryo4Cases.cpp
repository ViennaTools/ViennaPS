#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <geometries/psMakeTrench.hpp>
#include <models/psHFCryoEtching.hpp>
#include <process/psProcess.hpp>
#include <psUtil.hpp>

#include <lsToMultiSurfaceMesh.hpp>
#include <lsVTKWriter.hpp>

using namespace viennaps;

static constexpr double gridDelta   = 1.0;
static constexpr double xExtent     = 80.0;
static constexpr double yExtent     = 2000.0;
static constexpr double trenchWidth = 10.0;
static constexpr double maskHeight  = 80.0;
static constexpr double processTime = 400.0;

// Writes the surface mesh and returns the etch depth: how far the deepest
// point of the etch front sits below the original substrate surface (y = 0).
static double saveMeshAndDepth(SmartPointer<Domain<double, 2>> domain,
                               const std::string &filename, double T) {
  auto mesh = viennals::Mesh<double>::New();
  viennals::ToMultiSurfaceMesh<double, 2>(domain->getLevelSets(), mesh).apply();

  const size_t n = mesh->nodes.size();
  mesh->getPointData().insertNextScalarData(
      std::vector<double>(n, T), "Temperature_K");
  viennals::VTKWriter<double>(mesh, filename).apply();

  double minY = 0.0;
  for (size_t i = 0; i < n; ++i)
    minY = std::min(minY, mesh->nodes[i][1]);
  return -minY;
}

static double runCase(double T, bool useDiffusion) {
  auto domain = SmartPointer<Domain<double, 2>>::New(gridDelta, xExtent, yExtent);
  MakeTrench<double, 2>(domain, trenchWidth, 0.0, 0.0,
                        maskHeight, 0.0, false, Material::SiO2).apply();

  HFCryoParameters<double> params;
  params.temperature               = T;
  params.ionFlux                   = 3.0;
  params.etchantFlux               = 5.0e3;
  params.gamma_HF                  = 0.9;
  params.Desorption.nu0            = 1.0e8;
  params.Desorption.E_des          = 0.20;
  params.Reaction.A_r              = 3.0e2;
  params.Reaction.E_a              = 0.10;
  params.DirectReaction.A_r        = 1.0e4;
  params.DirectReaction.E_a        = 0.25;
  params.IonActivation.A_act       = 1.0;
  params.Diffusion.D0              = 1.0e3;
  params.Diffusion.omega           = 0.25;
  params.Ions.meanEnergy           = 150.0;
  params.Ions.sigmaEnergy          = 10.0;
  params.Ions.exponent             = 600.0;
  params.Config.useTemperatureDependence = true;
  params.Config.usePhysisorption         = true;
  params.Config.useSurfaceDiffusion      = useDiffusion;  // toggled per run
  params.Config.T_ref                    = 300.0;

  auto model = SmartPointer<HFCryoEtching<double, 2>>::New(params);
  Process<double, 2> process(domain, model, processTime);

  RayTracingParameters rayParams;
  rayParams.raysPerPoint = 5000;
  process.setParameters(rayParams);

  AdvectionParameters advParams;
  advParams.spatialScheme  = SpatialScheme::WENO_3RD_ORDER;
  advParams.temporalScheme = TemporalScheme::RUNGE_KUTTA_2ND_ORDER;
  process.setParameters(advParams);

  process.apply();

  const std::string tag   = useDiffusion ? "diffON" : "diffOFF";
  const std::string fname =
      "HFCryo_" + tag + "_T" + std::to_string((int)T) + "K.vtp";
  return saveMeshAndDepth(domain, fname, T);
}

int main() {
  Logger::setLogLevel(LogLevel::WARNING);
  omp_set_num_threads(4);

  units::Length::setUnit(units::Length::NANOMETER);
  units::Time::setUnit(units::Time::SECOND);

  const double Tlist[4] = {150., 200., 250., 300.};

  std::cout << "=== HF Cryo Etching: surface diffusion OFF vs ON ===\n";
  std::cout << "Geometry: " << trenchWidth << "nm trench, " << maskHeight
            << "nm mask, " << processTime << "s process\n\n";

  double depthOff[4], depthOn[4];

  std::cout << "--- Surface diffusion OFF ---\n";
  for (int i = 0; i < 4; ++i) {
    std::cout << "  T=" << std::setw(3) << (int)Tlist[i] << "K ... "
              << std::flush;
    depthOff[i] = runCase(Tlist[i], false);
    std::cout << "depth = " << std::fixed << std::setprecision(1)
              << depthOff[i] << " nm\n";
  }

  std::cout << "\n--- Surface diffusion ON ---\n";
  for (int i = 0; i < 4; ++i) {
    std::cout << "  T=" << std::setw(3) << (int)Tlist[i] << "K ... "
              << std::flush;
    depthOn[i] = runCase(Tlist[i], true);
    std::cout << "depth = " << std::fixed << std::setprecision(1)
              << depthOn[i] << " nm\n";
  }

  std::cout << "\n=== Summary: trench etch depth (nm) ===\n";
  std::cout << "   T[K]    diffOFF     diffON      gain\n";
  for (int i = 0; i < 4; ++i) {
    const double gain =
        depthOff[i] > 1e-9 ? (depthOn[i] / depthOff[i] - 1.0) * 100.0 : 0.0;
    std::cout << "   " << std::setw(4) << (int)Tlist[i] << "   " << std::setw(8)
              << std::setprecision(1) << depthOff[i] << "   " << std::setw(8)
              << depthOn[i] << "   " << std::setw(7) << std::setprecision(1)
              << gain << "%\n";
  }

  std::cout << "\n=== Done: 8 VTP files (HFCryo_diffOFF/ON_T*.vtp) ===\n";
  return 0;
}
