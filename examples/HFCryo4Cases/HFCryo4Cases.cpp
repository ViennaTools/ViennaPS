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
static constexpr double yExtent     = 2000.0;  // 700 → 2000 (깊은 트렌치 수용)
static constexpr double trenchWidth = 10.0;
static constexpr double maskHeight  = 80.0;    // 50 → 80 (두꺼운 마스크)
static constexpr double processTime = 400.0;   // 90 → 400 (긴 공정 시간)

// Save mesh with Temperature_K and CaseID as point scalars.
// In ParaView: Color by "Temperature_K" → temperature gradient (150-300K)
//              Color by "CaseID"         → 4 distinct case colors (0-3)
static void saveColoredMesh(SmartPointer<Domain<double, 2>> domain,
                             const std::string              &filename,
                             double T, int caseId) {
  auto mesh = viennals::Mesh<double>::New();
  viennals::ToMultiSurfaceMesh<double, 2>(domain->getLevelSets(), mesh).apply();

  const size_t n = mesh->nodes.size();
  mesh->getPointData().insertNextScalarData(
      std::vector<double>(n, T),            "Temperature_K");
  mesh->getPointData().insertNextScalarData(
      std::vector<double>(n, double(caseId)), "CaseID");

  viennals::VTKWriter<double>(mesh, filename).apply();
}

static void runCase(const std::string              &filename,
                    HFCryoParameters<double>        params,
                    int                             caseId) {
  auto domain = SmartPointer<Domain<double, 2>>::New(gridDelta, xExtent, yExtent);
  MakeTrench<double, 2>(domain, trenchWidth, 0.0, 0.0,
                         maskHeight, 0.0, false, Material::SiO2).apply();

  auto model = SmartPointer<HFCryoEtching<double, 2>>::New(params);
  Process<double, 2> process(domain, model, processTime);

  RayTracingParameters rayParams;
  rayParams.raysPerPoint = 5000;  // 3000 → 5000 (깊은 트렌치 정확도↑)
  process.setParameters(rayParams);

  AdvectionParameters advParams;
  advParams.spatialScheme  = SpatialScheme::WENO_3RD_ORDER;
  advParams.temporalScheme = TemporalScheme::RUNGE_KUTTA_2ND_ORDER;
  process.setParameters(advParams);

  process.apply();

  saveColoredMesh(domain, filename + ".vtp", params.temperature, caseId);
  std::cout << "done -> " << filename << ".vtp\n";
}

int main() {
  Logger::setLogLevel(LogLevel::WARNING);
  omp_set_num_threads(4);

  units::Length::setUnit(units::Length::NANOMETER);
  units::Time::setUnit(units::Time::SECOND);

  // ── Base parameters ─────────────────────────────────────────────────────────
  HFCryoParameters<double> base;
  base.ionFlux             = 3.0;     // 1.0 → 3.0  (이온 강화 → 수직 식각↑)
  base.etchantFlux         = 5.0e3;  // 1e3 → 5e3  (HF 공급↑)
  base.gamma_HF            = 0.9;
  base.Desorption.nu0      = 1.0e8;
  base.Desorption.E_des    = 0.20;
  base.Reaction.A_r        = 3.0e2;
  base.Reaction.E_a        = 0.10;
  base.DirectReaction.A_r  = 1.0e4;
  base.DirectReaction.E_a  = 0.25;
  base.IonActivation.A_act = 1.0;
  base.Diffusion.D0        = 0.0;
  base.Diffusion.omega     = 0.25;
  base.Ions.meanEnergy     = 150.0;  // 100 → 150 eV (더 높은 이온 에너지)
  base.Ions.sigmaEnergy    = 10.0;
  base.Ions.exponent       = 600.0;  // 300 → 600  (이온빔 더 수직, 이방성↑)
  base.Config.T_ref        = 300.0;

  // ── Case definitions ─────────────────────────────────────────────────────────
  struct CaseInfo {
    std::string tag;
    std::string desc;
    bool        tempDep;
    bool        phys;
    bool        diff;
    double      D0;
  };

  const CaseInfo cases[4] = {
    {"C0_none", "nothing (constant k_des, single theta, no diffusion)",
     false, false, false, 0.0},
    {"C1_temp", "+Arrhenius k_des (single theta, no diffusion)",
     true,  false, false, 0.0},
    {"C2_phys", "+physisorption (theta_phys+theta_chem, no diffusion)",
     true,  true,  false, 0.0},
    {"C3_diff", "+surface diffusion (D0=1e3)",
     true,  true,  true,  1.0e3},
  };

  const double Tlist[4] = {150., 200., 250., 300.};

  // ── Print rate table ─────────────────────────────────────────────────────────
  std::cout << "=== HF Cryo 4-Case x 4-Temperature Comparison ===\n";
  std::cout << "Geometry: " << trenchWidth << "nm trench, "
            << processTime << "s process\n\n";

  std::cout << "Effective k_des per case/temperature:\n";
  std::cout << std::setw(20) << "" << std::setw(10) << "150K"
            << std::setw(10) << "200K" << std::setw(10) << "250K"
            << std::setw(10) << "300K\n";
  std::cout << std::string(60, '-') << "\n";
  for (const auto &ci : cases) {
    std::cout << std::setw(20) << ci.tag;
    for (double T : Tlist) {
      auto p = base;
      p.temperature = T;
      p.Config.useTemperatureDependence = ci.tempDep;
      std::cout << std::setw(10) << std::setprecision(1) << std::fixed
                << p.effective_k_des();
    }
    std::cout << "\n";
  }
  std::cout << "\n";

  // ── Run all 16 simulations ───────────────────────────────────────────────────
  for (int c = 0; c < 4; ++c) {
    const auto &ci = cases[c];
    std::cout << "=== Case " << c << ": " << ci.desc << " ===\n";

    for (double T : Tlist) {
      std::cout << "  T=" << std::setw(3) << (int)T << "K ... ";
      std::cout.flush();

      auto p = base;
      p.temperature                     = T;
      p.Config.useTemperatureDependence = ci.tempDep;
      p.Config.usePhysisorption         = ci.phys;
      p.Config.useSurfaceDiffusion      = ci.diff;
      p.Diffusion.D0                    = ci.D0;

      std::string fname = ci.tag + "_T" + std::to_string((int)T) + "K";
      runCase(fname, p, c);
    }
    std::cout << "\n";
  }

  std::cout << "=== Done: 16 VTP files generated ===\n\n";
  std::cout << "ParaView: open all *.vtp files\n";
  std::cout << "  Color by 'Temperature_K' -> blue(150K) to red(300K)\n";
  std::cout << "  Color by 'CaseID'        -> 4 discrete case colors\n";

  return 0;
}
