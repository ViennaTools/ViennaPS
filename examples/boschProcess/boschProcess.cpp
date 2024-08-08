#include <geometries/psMakeHole.hpp>
#include <models/psIsotropicProcess.hpp>
#include <models/psSF6O2Etching.hpp>
#include <psProcess.hpp>
#include <psUtils.hpp>

namespace ps = viennaps;

template <class NumericType, int D>
void etch(ps::SmartPointer<ps::Domain<NumericType, D>> domain,
          ps::utils::Parameters &params) {
  auto model = ps::SmartPointer<ps::SF6Etching<NumericType, D>>::New(
      params.get("ionFlux"), params.get("etchantFlux"),
      params.get("meanEnergy"), params.get("sigmaEnergy"),
      params.get("ionExponent"));
  ps::Process<NumericType, D>(domain, model, params.get("etchTime")).apply();
}

template <class NumericType, int D>
void deposit(ps::SmartPointer<ps::Domain<NumericType, D>> domain,
             NumericType time, NumericType rate) {
  domain->duplicateTopLevelSet(ps::Material::Polymer);
  auto model =
      ps::SmartPointer<ps::IsotropicProcess<NumericType, D>>::New(rate);
  ps::Process<NumericType, D>(domain, model, time).apply();
}

template <class NumericType, int D>
void ash(ps::SmartPointer<ps::Domain<NumericType, D>> domain) {
  domain->removeTopLevelSet();
}

int main(int argc, char **argv) {
  constexpr int D = 3;
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
  ps::MakeHole<NumericType, D>(
      geometry, params.get("gridDelta") /* grid delta */,
      params.get("xExtent") /*x extent*/, params.get("yExtent") /*y extent*/,
      params.get("holeRadius") /*hole radius*/,
      params.get("maskHeight") /* mask height*/, 0., 0 /* base height */,
      false /* periodic boundary */, true /*create mask*/, ps::Material::Si)
      .apply();

  const NumericType depositionRate = params.get("depositionRate");
  const NumericType depositionTime = params.get("depositionTime");
  const int numCycles = params.get<int>("numCycles");

  int n = 0;
  etch(geometry, params);
  for (int i = 0; i < numCycles; ++i) {
    geometry->saveSurfaceMesh("boschProcess_" + std::to_string(n++) + ".vtp");
    deposit(geometry, depositionTime, depositionRate);
    geometry->saveSurfaceMesh("boschProcess_" + std::to_string(n++) + ".vtp");
    etch(geometry, params);
    geometry->saveSurfaceMesh("boschProcess_" + std::to_string(n++) + ".vtp");
    ash(geometry);
  }

  geometry->saveSurfaceMesh("boschProcess_final.vtp");
}