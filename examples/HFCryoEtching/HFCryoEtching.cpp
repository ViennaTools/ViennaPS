#include <geometries/psMakeTrench.hpp>
#include <models/psHFCryoEtching.hpp>
#include <process/psProcess.hpp>
#include <psUtil.hpp>

using namespace viennaps;

int main() {
  using NumericType = double;
  constexpr int D = 2;

  Logger::setLogLevel(LogLevel::INFO);
  omp_set_num_threads(4);

  // Units
  units::Length::setUnit(units::Length::Nanometer);
  units::Time::setUnit(units::Time::Second);

  // Geometry: SiO2 trench with mask
  NumericType gridDelta  = 1.0;   // nm
  NumericType xExtent    = 60.0;  // nm
  NumericType yExtent    = 60.0;  // nm
  NumericType trenchWidth = 20.0; // nm
  NumericType maskHeight  = 20.0; // nm

  auto domain = SmartPointer<Domain<NumericType, D>>::New(
      gridDelta, xExtent, yExtent);

  // Fill substrate with SiO2, mask with Mask material
  MakeTrench<NumericType, D>(domain,
                              trenchWidth,
                              0.0,        // trench depth (0 = flat start)
                              0.0,        // taper angle
                              maskHeight,
                              0.0,        // mask taper angle
                              false)      // periodic boundary
      .apply();

  // Replace default Si substrate material with SiO2
  domain->setMaterial(Material::SiO2);

  // HF cryo etching parameters
  HFCryoParameters<NumericType> params;
  params.ionFlux      = 1.0;     // 1e15 /cm²/s
  params.etchantFlux  = 1.0e3;   // 1e15 /cm²/s
  params.temperature  = 200.;    // K  (cryogenic ~-73°C)
  params.gamma_HF     = 0.9;     // HF sticking probability on bare SiO2

  // Frenkel-Arrhenius desorption
  params.Desorption.nu0   = 1.0e13; // 1/s
  params.Desorption.E_des = 0.25;   // eV

  // Arrhenius reaction rate
  params.Reaction.A_r = 3.0e2;  // 1e15 cm⁻²s⁻¹
  params.Reaction.E_a = 0.10;   // eV

  // Ion properties
  params.Ions.meanEnergy  = 100.; // eV
  params.Ions.sigmaEnergy = 10.;  // eV
  params.Ions.exponent    = 300.; // angular distribution

  auto model = SmartPointer<HFCryoEtching<NumericType, D>>::New(params);

  // Print computed Arrhenius rates at the set temperature
  std::cout << "Temperature : " << params.temperature << " K\n";
  std::cout << "k_des(T)   : " << params.k_des() << " /s\n";
  std::cout << "k_r(T)     : " << params.k_r()   << " (1e15 cm⁻²s⁻¹)\n";

  // Process
  NumericType processTime = 30.; // seconds

  Process<NumericType, D> process(domain, model, processTime);

  RayTracingParameters rayParams;
  rayParams.raysPerPoint = 3000;
  process.setParameters(rayParams);

  AdvectionParameters advParams;
  advParams.spatialScheme  = SpatialScheme::WENO_3RD_ORDER;
  advParams.temporalScheme = TemporalScheme::RUNGE_KUTTA_2ND_ORDER;
  process.setParameters(advParams);

  // Save initial surface
  domain->saveSurfaceMesh("HFCryo_initial.vtp");

  std::cout << "Running HF cryo etching simulation...\n";
  process.apply();

  // Save final surface
  domain->saveSurfaceMesh("HFCryo_final.vtp");
  std::cout << "Done. Output: HFCryo_final.vtp\n";

  // --- Temperature sweep: compare 150 K vs 200 K vs 250 K ---
  std::cout << "\n--- Temperature effect on k_r ---\n";
  for (NumericType T : {150., 200., 250., 300.}) {
    params.temperature = T;
    std::cout << "T=" << T << " K  k_r=" << params.k_r()
              << "  k_des=" << params.k_des() << "\n";
  }

  return 0;
}
