#include <geometries/psMakeTrench.hpp>
#include <models/psDirectionalEtching.hpp>
#include <models/psIsotropicProcess.hpp>
#include <psProcess.hpp>
#include <psUtils.hpp>

namespace ps = viennaps;

template <class NumericType, int D>
void etch(ps::SmartPointer<ps::Domain<NumericType, D>> domain,
          ps::utils::Parameters &params) {
  typename ps::DirectionalEtching<NumericType, D>::RateSet rateSet;
  rateSet.direction = {0.};
  rateSet.direction[D - 1] = -1.;
  rateSet.directionalVelocity = params.get("ionRate");
  rateSet.isotropicVelocity = params.get("neutralRate");
  rateSet.maskMaterials =
      std::vector<ps::Material>{ps::Material::Mask, ps::Material::Polymer};

  auto etchModel =
      ps::SmartPointer<ps::DirectionalEtching<NumericType, D>>::New(rateSet);
  ps::Process<NumericType, D>(domain, etchModel, params.get("etchTime"))
      .apply();
}

template <class NumericType, int D>
void punchThrough(ps::SmartPointer<ps::Domain<NumericType, D>> domain,
                  ps::utils::Parameters &params) {
  typename ps::DirectionalEtching<NumericType, D>::RateSet rateSet;
  rateSet.direction = {0.};
  rateSet.direction[D - 1] = -1.;
  rateSet.directionalVelocity =
      params.get("depositionThickness") + params.get("gridDelta") / 2.;
  rateSet.isotropicVelocity = 0.;
  rateSet.maskMaterials = std::vector<ps::Material>{ps::Material::Mask};

  // punch through step
  auto depoRemoval =
      ps::SmartPointer<ps::DirectionalEtching<NumericType, D>>::New(rateSet);
  ps::Process<NumericType, D>(domain, depoRemoval, 1.).apply();
}

template <class NumericType, int D>
void deposit(ps::SmartPointer<ps::Domain<NumericType, D>> domain,
             NumericType depositionThickness) {
  domain->duplicateTopLevelSet(ps::Material::Polymer);
  auto model = ps::SmartPointer<ps::IsotropicProcess<NumericType, D>>::New(
      depositionThickness);
  ps::Process<NumericType, D>(domain, model, 1.).apply();
}

template <class NumericType, int D>
void ash(ps::SmartPointer<ps::Domain<NumericType, D>> domain) {
  domain->removeTopLevelSet();
}

int main(int argc, char **argv) {
  constexpr int D = 2;
  using NumericType = double;

  ps::Logger::setLogLevel(ps::LogLevel::INFO);
  omp_set_num_threads(16);

  // Parse the parameters
  ps::utils::Parameters params;
  if (argc > 1) {
    params.readConfigFile(argv[1]);
  } else {
    std::cout << "Usage: " << argv[0] << " <config file>" << std::endl;
    return 1;
  }

  // geometry setup
  auto geometry = ps::SmartPointer<ps::Domain<NumericType, D>>::New();
  ps::MakeTrench<NumericType, D>(
      geometry, params.get("gridDelta") /* grid delta */,
      params.get("xExtent") /*x extent*/, params.get("yExtent") /*y extent*/,
      params.get("trenchWidth") /*hole radius*/,
      params.get("maskHeight") /* mask height*/, 0., 0 /* base height */,
      false /* periodic boundary */, true /*create mask*/, ps::Material::Si)
      .apply();

  const NumericType depositionThickness = params.get("depositionThickness");
  const int numCycles = params.get<int>("numCycles");

  int n = 0;
  etch(geometry, params);
  for (int i = 0; i < numCycles; ++i) {
    geometry->saveSurfaceMesh("boschProcess_" + std::to_string(n++) + ".vtp");
    deposit(geometry, depositionThickness);
    geometry->saveSurfaceMesh("boschProcess_" + std::to_string(n++) + ".vtp");
    punchThrough(geometry, params);
    geometry->saveSurfaceMesh("boschProcess_" + std::to_string(n++) + ".vtp");
    etch(geometry, params);
    geometry->saveSurfaceMesh("boschProcess_" + std::to_string(n++) + ".vtp");
    ash(geometry);
  }
  geometry->saveSurfaceMesh("boschProcess_" + std::to_string(n++) + ".vtp");
  geometry->saveVolumeMesh("boschProcess_final");
}