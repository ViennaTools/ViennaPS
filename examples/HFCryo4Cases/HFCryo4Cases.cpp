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

static void saveMesh(SmartPointer<Domain<double, 2>> domain,
                     const std::string &filename, double T) {
  auto mesh = viennals::Mesh<double>::New();
  viennals::ToMultiSurfaceMesh<double, 2>(domain->getLevelSets(), mesh).apply();

  const size_t n = mesh->nodes.size();
  mesh->getPointData().insertNextScalarData(
      std::vector<double>(n, T), "Temperature_K");

  viennals::VTKWriter<double>(mesh, filename).apply();
}

static void runCase(double T) {
  std::cout << "  T=" << std::setw(3) << (int)T << "K ... ";
  std::cout.flush();

  auto domain = SmartPointer<Domain<double, 2>>::New(gridDelta, xExtent, yExtent);
  MakeTrench<double, 2>(domain, trenchWidth, 0.0, 0.0,
                        maskHeight, 0.0, false, Material::SiO2).apply();

  // ── Parameters: all 3 features ON ───────────────────────────────────────────
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
  params.Diffusion.D0              = 1.0e3;   // surface diffusion ON
  params.Diffusion.omega           = 0.25;
  params.Ions.meanEnergy           = 150.0;
  params.Ions.sigmaEnergy          = 10.0;
  params.Ions.exponent             = 600.0;
  params.Config.useTemperatureDependence = true;  // feature 1: Arrhenius
  params.Config.usePhysisorption         = true;  // feature 2: physisorption
  params.Config.useSurfaceDiffusion      = true;  // feature 3: surface diffusion
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

  const std::string fname = "HFCryo_T" + std::to_string((int)T) + "K.vtp";
  saveMesh(domain, fname, T);
  std::cout << "done -> " << fname << "\n";
}

int main() {
  Logger::setLogLevel(LogLevel::WARNING);
  omp_set_num_threads(4);

  units::Length::setUnit(units::Length::NANOMETER);
  units::Time::setUnit(units::Time::SECOND);

  std::cout << "=== HF Cryo Etching: full model (Arrhenius + physisorption + surface diffusion) ===\n";
  std::cout << "Geometry: " << trenchWidth << "nm trench, "
            << maskHeight << "nm mask, " << processTime << "s process\n\n";

  const double Tlist[4] = {150., 200., 250., 300.};

  for (double T : Tlist)
    runCase(T);

  std::cout << "\n=== Done: 4 VTP files generated ===\n";
  std::cout << "ParaView: open HFCryo_T*.vtp, Color by 'Temperature_K'\n";

  return 0;
}
