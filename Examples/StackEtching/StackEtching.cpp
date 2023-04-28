#include <FluorocarbonEtching.hpp>
#include <psMakeStack.hpp>
#include <psProcess.hpp>
#include <psToSurfaceMesh.hpp>
#include <psUtils.hpp>
#include <psWriteVisualizationMesh.hpp>

#include "Parameters.hpp"

psSmartPointer<psMaterialMap> createMaterialMap(const int numLayers) {
  auto matMap = psSmartPointer<psMaterialMap>::New();
  matMap->insertNextMaterial(psMaterial::Mask); // mask
  matMap->insertNextMaterial(psMaterial::Si);   // substrate
  for (int i = 0; i < numLayers; i++) {
    if (i % 2 == 0) {
      matMap->insertNextMaterial(psMaterial::SiO2);
    } else {
      matMap->insertNextMaterial(psMaterial::Si3N4);
    }
  }
  //   matMap->insertNextMaterial(4); // depolayer
  return matMap;
}

int main(int argc, char *argv[]) {
  using NumericType = double;
  constexpr int D = 2;

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

  auto domain = psSmartPointer<psDomain<NumericType, D>>::New();
  psMakeStack<NumericType, D>(domain, params.gridDelta, params.xExtent,
                              params.yExtent, params.numLayers,
                              params.layerHeight, params.substrateHeight,
                              params.holeRadius, params.maskHeight, false)
      .apply();

  // copy top layer for deposition
  auto depoLayer = psSmartPointer<lsDomain<NumericType, D>>::New(
      domain->getLevelSets()->back());
  domain->insertNextLevelSetAsMaterial(depoLayer, psMaterial::Polymer);

  psWriteVisualizationMesh<NumericType, D>(domain, "initial").apply();
}