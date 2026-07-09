// CF4/Ar silicon etching benchmark example.
//
// Uses the dedicated CF4/Ar-on-Si model (CF4ArEtching), a thin configuration of
// the generic ViennaPS PlasmaEtching framework. It therefore runs on both the
// CPU and the GPU ray tracer.
//
//   Ion         -> Ar+           (ionFlux)
//   Etchant     -> F   radical   (etchantFlux),     coverage theta_F
//   Passivation -> lumped CFx    (passivationFlux), coverage theta_CF
//
//   Step 1 : F + Ar+          -> passivationFlux = 0
//   Step 2 : add lumped CFx   -> passivationFlux > 0 (sidewall passivation)
//
// Geometry: set D = 2 for a masked Si trench, D = 3 for a masked Si
// cylindrical hole.

#include <geometries/psMakeHole.hpp>
#include <geometries/psMakeTrench.hpp>
#include <models/psCF4ArEtching.hpp>

#include <process/psProcess.hpp>
#include <psUtil.hpp>

using namespace viennaps;

int main() {
  using NumericType = double;
  constexpr int D = 2; // 2 -> trench, 3 -> cylindrical hole

  Logger::setLogLevel(LogLevel::INFO);
  omp_set_num_threads(16);

  // feature-scale units
  units::Length::setUnit("nm");
  units::Time::setUnit("s");

  // geometry parameters (all lengths in nm)
  const NumericType gridDelta = 2.0;
  const NumericType xExtent = 200.0;
  const NumericType yExtent = 200.0;
  const NumericType featureWidth = 80.0; // trench width / hole diameter
  const NumericType featureDepth = 40.0;
  const NumericType maskHeight = 60.0;
  const NumericType processTime = 6.0; // seconds (~14 nm/s blanket Si rate)

  // Prefer the GPU triangle-mesh flux engine when compiled with GPU support.
#ifdef VIENNACORE_COMPILE_GPU
  const auto fluxEngine = FluxEngineType::GPU_TRIANGLE;
#else
  const auto fluxEngine = FluxEngineType::CPU_DISK;
#endif

  auto makeGeometry = [&]() {
    auto geometry =
        Domain<NumericType, D>::New(gridDelta, xExtent, yExtent);
    if constexpr (D == 2) {
      MakeTrench<NumericType, D>(geometry, featureWidth, featureDepth,
                                 0.0, // trenchTaperAngle
                                 maskHeight,
                                 0.0,   // maskTaperAngle
                                 false, // halfTrench
                                 Material::Si, Material::Mask)
          .apply();
    } else {
      // 3D cylindrical hole (quarter geometry exploits the 4-fold symmetry).
      MakeHole<NumericType, D>(geometry, featureWidth / 2.0, featureDepth,
                               0.0, // holeTaperAngle
                               maskHeight,
                               0.0, // maskTaperAngle
                               HoleShape::QUARTER, Material::Si, Material::Mask)
          .apply();
    }
    return geometry;
  };

  auto runSimulation = [&](PlasmaEtchingParameters<NumericType> params,
                           const std::string &name) {
    auto geometry = makeGeometry();
    auto model =
        SmartPointer<CF4ArEtching<NumericType, D>>::New(params);

    CoverageParameters coverageParams;
    coverageParams.tolerance = 1e-4;

    RayTracingParameters rayTracingParams;
    rayTracingParams.raysPerPoint = 1000;

    Process<NumericType, D> process(geometry, model);
    process.setProcessDuration(processTime);
    process.setParameters(coverageParams);
    process.setParameters(rayTracingParams);
    process.setFluxEngineType(fluxEngine);

    std::cout << "Running: " << name << std::endl;
    process.apply();
    geometry->saveSurfaceMesh("CF4Ar_" + name + ".vtp", true);
  };

  makeGeometry()->saveSurfaceMesh("CF4Ar_initial.vtp");

  // Step 1: F + Ar+ baseline
  auto step1 = CF4ArEtching<NumericType, D>::defaultParameters();
  step1.ionFlux = 12.0;
  step1.etchantFlux = 1.8e2; // reduced for a moderate rate (~14 nm/s blanket)
  step1.passivationFlux = 0.0;
  step1.Ions.meanEnergy = 100.0;
  step1.Ions.sigmaEnergy = 10.0;
  step1.Ions.exponent = 500.0;
  runSimulation(step1, "step1_F_Ar");

  // Step 2: add lumped CFx residue (sidewall passivation)
  auto step2 = step1;
  step2.passivationFlux = 1.0e2;
  step2.Passivation.A_ie = 3.0;
  step2.Passivation.Eth_ie = 10.0;
  runSimulation(step2, "step2_CFx");

  // Step 1 + surface diffusion of the fluorine coverage theta_F.
  // A positive coefficient enables operator-split diffusion of the "eCoverage"
  // (theta_F) field after each local reaction update.
  auto step1Diff = step1;
  step1Diff.etchantDiffusionCoefficient = 5.0e3; // D_F in nm^2/s
  runSimulation(step1Diff, "step1_F_diffusion");
}
