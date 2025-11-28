#include <geometries/psMakeTrench.hpp>
#include <models/psDirectionalProcess.hpp>
#include <models/psGeometricDistributionModels.hpp>
#include <models/psIsotropicProcess.hpp>
#include <process/psProcess.hpp>
#include <psUtil.hpp>

using namespace viennaps;
constexpr int D = 2;
using NumericType = double;

void etch(SmartPointer<Domain<NumericType, D>> &domain,
          util::Parameters &params) {
  std::cout << "  - Etching - " << std::endl;
  DirectionalProcess<NumericType, D>::RateSet rateSet;
  rateSet.direction = Vec3D<NumericType>{0.};
  rateSet.direction[D - 1] = -1.;
  rateSet.directionalVelocity = params.get("ionRate");
  rateSet.isotropicVelocity = params.get("neutralRate");
  rateSet.maskMaterials =
      std::vector<Material>{Material::Mask, Material::Polymer};
  rateSet.calculateVisibility = true;

  auto etchModel =
      SmartPointer<DirectionalProcess<NumericType, D>>::New(rateSet);
  Process<NumericType, D>(domain, etchModel, params.get("etchTime")).apply();
}

void punchThrough(SmartPointer<Domain<NumericType, D>> &domain,
                  util::Parameters &params) {
  std::cout << "  - Punching through - " << std::endl;
  DirectionalProcess<NumericType, D>::RateSet rateSet;
  rateSet.direction = Vec3D<NumericType>{0.};
  rateSet.direction[D - 1] = -1.;
  rateSet.directionalVelocity =
      -(params.get("depositionThickness") + params.get("gridDelta") / 2.);
  rateSet.isotropicVelocity = 0.;
  rateSet.maskMaterials = std::vector<Material>{Material::Mask};
  rateSet.calculateVisibility = true;

  // punch through step
  auto depoRemoval =
      SmartPointer<DirectionalProcess<NumericType, D>>::New(rateSet);
  Process<NumericType, D>(domain, depoRemoval, 1.).apply();
}

void deposit(SmartPointer<Domain<NumericType, D>> &domain,
             NumericType depositionThickness) {
  std::cout << "  - Deposition - " << std::endl;
  domain->duplicateTopLevelSet(Material::Polymer);
  auto model = SmartPointer<SphereDistribution<NumericType, D>>::New(
      depositionThickness);
  Process<NumericType, D>(domain, model).apply();
}

void ash(SmartPointer<Domain<NumericType, D>> &domain) {
  domain->removeTopLevelSet();
  domain->removeStrayPoints();
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
  const std::string name = "boschProcessEmulate_";

  int n = 0;
  geometry->saveSurfaceMesh(name + std::to_string(n++) + ".vtp");
  etch(geometry, params);
  geometry->saveSurfaceMesh(name + std::to_string(n++) + ".vtp");

  for (int i = 0; i < numCycles; ++i) {
    std::cout << "Cycle " << i + 1 << std::endl;
    deposit(geometry, depositionThickness);
    geometry->saveSurfaceMesh(name + std::to_string(n++) + ".vtp");
    punchThrough(geometry, params);
    geometry->saveSurfaceMesh(name + std::to_string(n++) + ".vtp");
    etch(geometry, params);
    geometry->saveSurfaceMesh(name + std::to_string(n++) + ".vtp");
    ash(geometry);
    geometry->saveSurfaceMesh(name + std::to_string(n++) + ".vtp");
  }

  geometry->saveVolumeMesh("boschProcessEmulate_final");
}