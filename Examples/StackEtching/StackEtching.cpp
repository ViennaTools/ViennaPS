#include <FluorocarbonEtching.hpp>
#include <SimpleDeposition.hpp>
#include <psExtrude.hpp>
#include <psMakeStack.hpp>
#include <psProcess.hpp>
#include <psToDiskMesh.hpp>
#include <psToSurfaceMesh.hpp>
#include <psUtils.hpp>
#include <psWriteVisualizationMesh.hpp>

#include "Parameters.hpp"

int main(int argc, char *argv[]) {
  using NumericType = double;
  constexpr int D = 2;

  psLogger::setLogLevel(psLogLevel::DEBUG);

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

  auto model = psSmartPointer<FluorocarbonEtching<NumericType, D>>::New(
      params.totalIonFlux, params.totalEtchantFlux, params.totalPolymerFlux,
      params.rfBiasPower);

  psProcess<NumericType, D> process;
  process.setDomain(domain);
  process.setProcessModel(model);
  process.setProcessDuration(params.processTime);
  process.setMaxCoverageInitIterations(1);
  process.setSmoothFlux(true);
  process.apply();

  psWriteVisualizationMesh<NumericType, D>(domain, "final").apply();

  std::cout << "Extruding to 3D ..." << std::endl;
  auto extruded = psSmartPointer<psDomain<NumericType, 3>>::New();
  std::array<NumericType, 2> extrudeExtent = {-20., 20.};
  psExtrude<NumericType>(domain, extruded, extrudeExtent, 2).apply();

  std::cout << "Writing to surface" << std::endl;
  {
    auto mesh = psSmartPointer<lsMesh<NumericType>>::New();
    psToDiskMesh<NumericType, 3>(extruded, mesh).apply();
    lsVTKWriter<NumericType>(mesh, "Extruded_surface.vtp").apply();
  }

  psWriteVisualizationMesh<NumericType, 3>(extruded, "final_extruded").apply();
}