#include <cmath>
#include <iostream>
#include <string>

#include <geometries/psMakeTrench.hpp>
#include <models/psHFCryoEtching.hpp>
#include <process/psProcess.hpp>
#include <psUtil.hpp>

using namespace viennaps;

static void runCase(const std::string &label,
                    double T, double gridDelta,
                    double xExtent, double yExtent,
                    double trenchWidth, double maskHeight,
                    double processTime,
                    HFCryoParameters<double> params) {
  params.temperature = T;

  auto domain = SmartPointer<Domain<double, 2>>::New(gridDelta, xExtent, yExtent);
  MakeTrench<double, 2>(domain, trenchWidth, 0.0, 0.0,
                         maskHeight, 0.0, false, Material::SiO2).apply();

  auto model = SmartPointer<HFCryoEtching<double, 2>>::New(params);
  Process<double, 2> process(domain, model, processTime);

  RayTracingParameters rayParams;
  rayParams.raysPerPoint = 3000;
  process.setParameters(rayParams);

  AdvectionParameters advParams;
  advParams.spatialScheme  = SpatialScheme::WENO_3RD_ORDER;
  advParams.temporalScheme = TemporalScheme::RUNGE_KUTTA_2ND_ORDER;
  process.setParameters(advParams);

  process.apply();

  std::string tag = label + "_T" + std::to_string((int)T) + "K";
  domain->saveSurfaceMesh(tag + ".vtp", true);
  domain->saveHullMesh(tag + "_hull.vtp");
  std::cout << "done -> " << tag << ".vtp\n";
}

int main() {
  Logger::setLogLevel(LogLevel::WARNING);
  omp_set_num_threads(4);

  units::Length::setUnit(units::Length::NANOMETER);
  units::Time::setUnit(units::Time::SECOND);

  const double gridDelta    = 1.0;
  const double xExtent      = 80.0;
  const double yExtent      = 700.0;
  const double trenchWidth  = 10.0;
  const double maskHeight   = 50.0;
  const double processTime  = 90.0;

  // Base parameters shared by both cases
  HFCryoParameters<double> params;
  params.ionFlux              = 1.0;
  params.etchantFlux          = 1.0e3;
  params.gamma_HF             = 0.9;
  params.Desorption.nu0       = 1.0e8;
  params.Desorption.E_des     = 0.20;
  params.Reaction.A_r         = 3.0e2;
  params.Reaction.E_a         = 0.10;
  params.IonActivation.A_act  = 1.0;
  params.Ions.meanEnergy      = 100.0;
  params.Ions.sigmaEnergy     = 10.0;
  params.Ions.exponent        = 300.0;

  // Print D_s per temperature
  std::cout << "=== HF Cryo Etching | Surface Diffusion Comparison ===\n\n";
  std::cout << "T(K)   k_des      k_r      k_r_direct   D_s(D0=1e3)\n";
  std::cout << "------------------------------------------------------\n";
  double Tlist[4] = {150.0, 200.0, 250.0, 300.0};
  for (int i = 0; i < 4; ++i) {
    params.temperature    = Tlist[i];
    params.Diffusion.D0   = 1.0e3;
    params.Diffusion.omega = 0.25;
    std::cout << Tlist[i] << "K  "
              << params.k_des()      << "  "
              << params.k_r()        << "  "
              << params.k_r_direct() << "  "
              << params.D_s()        << "\n";
  }
  std::cout << "\n";

  // Save initial geometry
  {
    auto domain = SmartPointer<Domain<double, 2>>::New(gridDelta, xExtent, yExtent);
    MakeTrench<double, 2>(domain, trenchWidth, 0.0, 0.0,
                           maskHeight, 0.0, false, Material::SiO2).apply();
    domain->saveSurfaceMesh("HFCryo_initial.vtp", true);
    std::cout << "Initial geometry saved.\n\n";
  }

  // ── Case A: No surface diffusion (D0=0) ──────────────────────────────────
  std::cout << "=== Case A: No Surface Diffusion (D0=0) ===\n";
  params.Diffusion.D0    = 0.0;
  params.Diffusion.omega = 0.25;
  for (int i = 0; i < 4; ++i) {
    std::cout << "  T=" << Tlist[i] << "K ... ";
    std::cout.flush();
    runCase("NoDiff", Tlist[i], gridDelta, xExtent, yExtent,
            trenchWidth, maskHeight, processTime, params);
  }

  // ── Case B: With surface diffusion (D0=1e3) ───────────────────────────────
  std::cout << "\n=== Case B: With Surface Diffusion (D0=1e3) ===\n";
  params.Diffusion.D0    = 1.0e3;
  params.Diffusion.omega = 0.25;
  for (int i = 0; i < 4; ++i) {
    std::cout << "  T=" << Tlist[i] << "K ... ";
    std::cout.flush();
    runCase("Diff", Tlist[i], gridDelta, xExtent, yExtent,
            trenchWidth, maskHeight, processTime, params);
  }

  std::cout << "\n=== Done ===\n";
  std::cout << "Compare in ParaView:\n";
  std::cout << "  NoDiff_T150K.vtp  vs  Diff_T150K.vtp\n";
  std::cout << "  NoDiff_T200K.vtp  vs  Diff_T200K.vtp\n";
  std::cout << "Color by MaterialIds to see Mask / SiO2\n";

  return 0;
}
