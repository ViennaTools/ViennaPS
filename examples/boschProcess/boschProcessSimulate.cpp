#include <geometries/psMakeTrench.hpp>
#include <models/psIsotropicProcess.hpp>
#include <models/psMultiParticleProcess.hpp>
#include <models/psSingleParticleProcess.hpp>
#include <psProcess.hpp>
#include <psUtil.hpp>

using namespace viennaps;
constexpr int D = 2;
using NumericType = double;

void etch(SmartPointer<Domain<NumericType, D>> domain,
          util::Parameters &params) {
  std::cout << "  - Etching - " << std::endl;
  auto etchModel = SmartPointer<MultiParticleProcess<NumericType, D>>::New();
  etchModel->addNeutralParticle(params.get("neutralStickingProbability"));
  etchModel->addIonParticle(params.get("ionSourceExponent"),
                            90. /*no reflections*/);
  const NumericType neutralRate = params.get("neutralRate");
  const NumericType ionRate = params.get("ionRate");
  etchModel->setRateFunction(
      [neutralRate, ionRate](const std::vector<NumericType> &fluxes,
                             const Material &material) {
        if (material == Material::Mask)
          return 0.;
        NumericType rate = fluxes[1] * ionRate;
        if (material == Material::Si)
          rate += fluxes[0] * neutralRate;
        return rate;
      });
  Process<NumericType, D>(domain, etchModel, params.get("etchTime")).apply();
}

void punchThrough(SmartPointer<Domain<NumericType, D>> domain,
                  util::Parameters &params) {
  std::cout << "  - Punching through - " << std::endl;
  NumericType depositionThickness = params.get("depositionThickness");

  // punch through step
  auto depoRemoval = SmartPointer<SingleParticleProcess<NumericType, D>>::New(
      -depositionThickness, 1. /* sticking */, params.get("ionSourceExponent"),
      Material::Mask);
  Process<NumericType, D>(domain, depoRemoval, 1.).apply();
}

void deposit(SmartPointer<Domain<NumericType, D>> domain,
             util::Parameters &params) {
  std::cout << "  - Deposition - " << std::endl;
  NumericType depositionThickness = params.get("depositionThickness");
  NumericType depositionSticking = params.get("depositionStickingProbability");
  domain->duplicateTopLevelSet(Material::Polymer);
  auto model = SmartPointer<SingleParticleProcess<NumericType, D>>::New(
      depositionThickness, depositionSticking);
  Process<NumericType, D>(domain, model, 1.).apply();
}

void ash(SmartPointer<Domain<NumericType, D>> domain) {
  domain->removeTopLevelSet();
}

void cleanup(SmartPointer<Domain<NumericType, D>> domain,
             NumericType threshold) {
  auto expand = SmartPointer<IsotropicProcess<NumericType, D>>::New(threshold);
  Process<NumericType, D>(domain, expand, 1.).apply();
  auto shrink = SmartPointer<IsotropicProcess<NumericType, D>>::New(-threshold);
  Process<NumericType, D>(domain, shrink, 1.).apply();
}

int main(int argc, char **argv) {

  Logger::setLogLevel(LogLevel::ERROR);
  omp_set_num_threads(16);

  // Parse the parameters
  util::Parameters params;
  if (argc > 1) {
    params.readConfigFile(argv[1]);
  } else {
    std::cout << "Usage: " << argv[0] << " <config file>" << std::endl;
    return 1;
  }

  // geometry setup
  auto geometry = Domain<NumericType, D>::New(
      params.get("gridDelta"), params.get("xExtent"), params.get("yExtent"));
  MakeTrench<NumericType, D>(geometry, params.get("trenchWidth"),
                             0.0, // trenchDepth
                             0.0, // trenchTaperAngle
                             params.get("maskHeight"))
      .apply();

  const NumericType depositionThickness = params.get("depositionThickness");
  const int numCycles = params.get<int>("numCycles");
  const std::string name = "boschProcessSimulate_";

  int n = 0;
  geometry->saveSurfaceMesh(name + std::to_string(n++) + ".vtp");
  etch(geometry, params);
  geometry->saveSurfaceMesh(name + std::to_string(n++) + ".vtp");

  for (int i = 0; i < numCycles; ++i) {
    std::cout << "Cycle " << i + 1 << std::endl;
    deposit(geometry, params);
    geometry->saveSurfaceMesh(name + std::to_string(n++) + ".vtp");
    punchThrough(geometry, params);
    geometry->saveSurfaceMesh(name + std::to_string(n++) + ".vtp");
    etch(geometry, params);
    geometry->saveSurfaceMesh(name + std::to_string(n++) + ".vtp");
    ash(geometry);
    geometry->saveSurfaceMesh(name + std::to_string(n++) + ".vtp");
  }

  cleanup(geometry, params.get("gridDelta"));
  geometry->saveVolumeMesh(name + "final");
}