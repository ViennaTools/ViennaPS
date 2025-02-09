#include <geometries/psMakeTrench.hpp>
#include <models/psIsotropicProcess.hpp>
#include <models/psMultiParticleProcess.hpp>
#include <models/psSingleParticleProcess.hpp>
#include <psProcess.hpp>
#include <psUtils.hpp>

using namespace viennaps;
constexpr int D = 2;
using NumericType = double;

void etch(SmartPointer<Domain<NumericType, D>> domain,
          utils::Parameters &params) {
  auto etchModel = SmartPointer<MultiParticleProcess<NumericType, D>>::New();
  etchModel->addNeutralParticle(params.get("neutralStickingProbability"));
  etchModel->addIonParticle(params.get("ionSourceExponent"));
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
                  utils::Parameters &params) {
  NumericType depositionThickness = params.get("depositionThickness");
  NumericType gridDelta = params.get("gridDelta");

  // punch through step
  auto depoRemoval = SmartPointer<SingleParticleProcess<NumericType, D>>::New(
      -(depositionThickness + gridDelta / 2.0), 1. /* sticking */,
      params.get("ionSourceExponent"), Material::Mask);
  Process<NumericType, D>(domain, depoRemoval, 1.).apply();
}

void deposit(SmartPointer<Domain<NumericType, D>> domain,
             NumericType depositionThickness) {
  domain->duplicateTopLevelSet(Material::Polymer);
  auto model =
      SmartPointer<IsotropicProcess<NumericType, D>>::New(depositionThickness);
  Process<NumericType, D>(domain, model, 1.).apply();
}

void ash(SmartPointer<Domain<NumericType, D>> domain) {
  domain->removeTopLevelSet();
}

int main(int argc, char **argv) {

  Logger::setLogLevel(LogLevel::INFO);
  omp_set_num_threads(16);

  // Parse the parameters
  utils::Parameters params;
  if (argc > 1) {
    params.readConfigFile(argv[1]);
  } else {
    std::cout << "Usage: " << argv[0] << " <config file>" << std::endl;
    return 1;
  }

  // geometry setup
  auto geometry = SmartPointer<Domain<NumericType, D>>::New();
  MakeTrench<NumericType, D>(
      geometry, params.get("gridDelta"), params.get("xExtent"),
      params.get("yExtent"), params.get("trenchWidth"),
      params.get("maskHeight"), 0. /* taper angle */, 0. /* base height */,
      false /* periodic boundary */, true /* create mask */, Material::Si)
      .apply();

  const NumericType depositionThickness = params.get("depositionThickness");
  const int numCycles = params.get<int>("numCycles");

  int n = 0;
  geometry->saveSurfaceMesh("boschProcess_" + std::to_string(n++) + ".vtp");
  etch(geometry, params);
  geometry->saveSurfaceMesh("boschProcess_" + std::to_string(n++) + ".vtp");

  for (int i = 0; i < numCycles; ++i) {
    deposit(geometry, depositionThickness);
    geometry->saveSurfaceMesh("boschProcess_" + std::to_string(n++) + ".vtp");
    punchThrough(geometry, params);
    geometry->saveSurfaceMesh("boschProcess_" + std::to_string(n++) + ".vtp");
    etch(geometry, params);
    geometry->saveSurfaceMesh("boschProcess_" + std::to_string(n++) + ".vtp");
    ash(geometry);
    geometry->saveSurfaceMesh("boschProcess_" + std::to_string(n++) + ".vtp");
  }

  geometry->saveVolumeMesh("boschProcess_final");
}