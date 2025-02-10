#include <geometries/psMakeTrench.hpp>
#include <models/psDirectionalEtching.hpp>
#include <models/psGeometricDistributionModels.hpp>
#include <models/psIsotropicProcess.hpp>
#include <psProcess.hpp>
#include <psUtils.hpp>

using namespace viennaps;
constexpr int D = 2;
using NumericType = double;

void etch(SmartPointer<Domain<NumericType, D>> domain,
          utils::Parameters &params) {
  typename DirectionalEtching<NumericType, D>::RateSet rateSet;
  rateSet.direction = {0.};
  rateSet.direction[D - 1] = -1.;
  rateSet.directionalVelocity = params.get("ionRate");
  rateSet.isotropicVelocity = params.get("neutralRate");
  rateSet.maskMaterials =
      std::vector<Material>{Material::Mask, Material::Polymer};
  rateSet.calculateVisibility = true;

  auto etchModel =
      SmartPointer<DirectionalEtching<NumericType, D>>::New(rateSet);
  Process<NumericType, D>(domain, etchModel, params.get("etchTime")).apply();
}

void punchThrough(SmartPointer<Domain<NumericType, D>> domain,
                  utils::Parameters &params) {
  typename DirectionalEtching<NumericType, D>::RateSet rateSet;
  rateSet.direction = {0.};
  rateSet.direction[D - 1] = -1.;
  rateSet.directionalVelocity =
      -(params.get("depositionThickness") + params.get("gridDelta") / 2.);
  rateSet.isotropicVelocity = 0.;
  rateSet.maskMaterials = std::vector<Material>{Material::Mask};
  rateSet.calculateVisibility = true;

  // punch through step
  auto depoRemoval =
      SmartPointer<DirectionalEtching<NumericType, D>>::New(rateSet);
  Process<NumericType, D>(domain, depoRemoval, 1.).apply();
}

void deposit(SmartPointer<Domain<NumericType, D>> domain,
             NumericType depositionThickness) {
  domain->duplicateTopLevelSet(Material::Polymer);
  auto model = SmartPointer<SphereDistribution<NumericType, D>>::New(
      depositionThickness, domain->getGridDelta());
  Process<NumericType, D>(domain, model).apply();
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
  geometry->saveSurfaceMesh("boschProcessEmulate_" + std::to_string(n++) +
                            ".vtp");
  etch(geometry, params);
  geometry->saveSurfaceMesh("boschProcessEmulate_" + std::to_string(n++) +
                            ".vtp");

  for (int i = 0; i < numCycles; ++i) {
    deposit(geometry, depositionThickness);
    geometry->saveSurfaceMesh("boschProcessEmulate_" + std::to_string(n++) +
                              ".vtp");
    punchThrough(geometry, params);
    geometry->saveSurfaceMesh("boschProcessEmulate_" + std::to_string(n++) +
                              ".vtp");
    etch(geometry, params);
    geometry->saveSurfaceMesh("boschProcessEmulate_" + std::to_string(n++) +
                              ".vtp");
    ash(geometry);
    geometry->saveSurfaceMesh("boschProcessEmulate_" + std::to_string(n++) +
                              ".vtp");
  }

  cleanup(geometry, params.get("gridDelta"));
  geometry->saveVolumeMesh("boschProcessEmulate_final");
}