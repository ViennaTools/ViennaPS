#include <geometries/psMakeTrench.hpp>
#include <models/psGeometricDistributionModels.hpp>

#include <psProcess.hpp>
#include <psUtils.hpp>

namespace ps = viennaps;

int main(int argc, char *argv[]) {
  using NumericType = double;
  static constexpr int D = 2;

  // Parse the parameters
  ps::utils::Parameters params;
  if (argc > 1) {
    params.readConfigFile(argv[1]);
  } else {
    std::cout << "Usage: " << argv[0] << " <config file>" << std::endl;
    return 1;
  }

  auto geometry = ps::SmartPointer<ps::Domain<NumericType, D>>::New();
  ps::MakeTrench<NumericType, D>(geometry, params.get("gridDelta"),
                                 params.get("xExtent"), params.get("yExtent"),
                                 params.get("trenchWidth"),
                                 params.get("trenchHeight"))
      .apply();

  // copy top layer to capture deposition
  geometry->duplicateTopLevelSet();

  auto model = ps::SmartPointer<ps::SphereDistribution<NumericType, D>>::New(
      params.get("layerThickness"), params.get("gridDelta"));

  ps::Process<NumericType, D> process;
  process.setDomain(geometry);
  process.setProcessModel(model);

  geometry->saveHullMesh("initial");

  process.apply();

  geometry->saveHullMesh("final");
}
