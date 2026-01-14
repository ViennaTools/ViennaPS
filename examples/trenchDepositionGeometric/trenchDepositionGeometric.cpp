#include <geometries/psMakeTrench.hpp>
#include <models/psGeometricDistributionModels.hpp>

#include <process/psProcess.hpp>
#include <psUtil.hpp>

namespace ps = viennaps;

int main(int argc, char *argv[]) {
  using NumericType = double;
  static constexpr int D = 2;

  // Parse the parameters
  ps::util::Parameters params;
  if (argc > 1) {
    params.readConfigFile(argv[1]);
  } else {
    // Try default config file
    params.readConfigFile("config.txt");
    if (params.m.empty()) {
      std::cout << "No configuration file provided!" << std::endl;
      std::cout << "Usage: " << argv[0] << " <config file>" << std::endl;
      return 1;
    }
  }

  auto geometry = ps::SmartPointer<ps::Domain<NumericType, D>>::New(
      params.get("gridDelta"), params.get("xExtent"), params.get("yExtent"));
  ps::MakeTrench<NumericType, D>(geometry, params.get("trenchWidth"),
                                 params.get("trenchHeight"),
                                 params.get("taperAngle"))
      .apply();

  // copy top layer to capture deposition
  geometry->duplicateTopLevelSet(ps::Material::SiO2);

  auto model = ps::SmartPointer<ps::SphereDistribution<NumericType, D>>::New(
      params.get("layerThickness"));

  ps::Process<NumericType, D> process;
  process.setDomain(geometry);
  process.setProcessModel(model);

  geometry->saveHullMesh("initial");

  process.apply();

  geometry->saveHullMesh("final");
}
