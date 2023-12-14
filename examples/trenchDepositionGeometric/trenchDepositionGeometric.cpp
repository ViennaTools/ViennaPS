#include <geometries/psMakeTrench.hpp>
#include <psGeometricDistributionModels.hpp>
#include <psProcess.hpp>
#include <psUtils.hpp>

#include "parameters.hpp"

int main(int argc, char *argv[]) {
  using NumericType = double;
  static constexpr int D = 2;

  // Parse the parameters
  Parameters<NumericType> params;
  if (argc > 1) {
    auto config = psUtils::readConfigFile(argv[1]);
    if (config.empty()) {
      std::cerr << "Empty config provided" << std::endl;
      return -1;
    }
    params.fromMap(config);
  }

  auto geometry = psSmartPointer<psDomain<NumericType, D>>::New();
  psMakeTrench<NumericType, D>(geometry, params.gridDelta, params.xExtent,
                               params.yExtent, params.trenchWidth,
                               params.trenchHeight)
      .apply();

  // copy top layer to capture deposition
  geometry->duplicateTopLevelSet();

  auto model = psSmartPointer<psSphereDistribution<NumericType, D>>::New(
      params.layerThickness, params.gridDelta);

  psProcess<NumericType, D> process;
  process.setDomain(geometry);
  process.setProcessModel(model);

  geometry->saveSurfaceMesh("initial.vtp");

  process.apply();

  geometry->saveSurfaceMesh("final.vtp");

  if constexpr (D == 2)
    geometry->saveVolumeMesh("final");
}
