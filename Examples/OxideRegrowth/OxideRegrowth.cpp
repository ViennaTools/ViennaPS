#include <csDenseCellSet.hpp>
#include <csTracing.hpp>

#include <psMakeStack.hpp>
#include <psProcess.hpp>
#include <psProcessModel.hpp>
#include <psWriteVisualizationMesh.hpp>

#include <StackRedeposition.hpp>

#include "Parameters.hpp"

int main(int argc, char **argv) {
  using NumericType = double;

  omp_set_num_threads(12);
  // Attention:
  // This model/example currently only works in 2D mode
  constexpr int D = 2;

  Parameters<NumericType> params;
  if (argc > 1) {
    auto config = psUtils::readConfigFile(argv[1]);
    if (config.empty()) {
      std::cerr << "Empty config provided" << std::endl;
      return -1;
    }
    params.fromMap(config);
  }

  const NumericType stability =
      2 * params.diffusionCoefficient /
      std::max(params.holeStreamVelocity, params.scallopStreamVelocity);
  std::cout << "Stability: " << stability << std::endl;
  if (0.5 * stability <= params.gridDelta) {
    std::cout << "Unstable parameters. Reduce grid spacing!" << std::endl;
    return -1;
  }

  auto domain = psSmartPointer<psDomain<NumericType, D>>::New();
  psMakeStack<NumericType, D>(domain, params.gridDelta, params.xExtent, 0.,
                              params.numLayers, params.layerHeight,
                              params.substrateHeight, params.holeRadius, false)
      .apply();
  // copy top layer for deposition
  auto depoLayer = psSmartPointer<lsDomain<NumericType, D>>::New(
      domain->getLevelSets()->back());
  domain->insertNextLevelSet(depoLayer);
  domain->generateCellSet(params.substrateHeight +
                              params.numLayers * params.layerHeight + 10.,
                          true /* true means cell set above surface */);
  auto &cellSet = domain->getCellSet();
  cellSet->writeVTU("initial.vtu");
  // we need neighborhood information for solving the
  // convection-diffusion equation on the cell set
  cellSet->buildNeighborhood();

  // The redeposition model captures byproducts from the selective etching
  // process in the cell set. The byproducts are then distributed by solving a
  // convection-diffusion equation on the cell set.
  auto redepoModel = psSmartPointer<RedepositionDynamics<NumericType, D>>::New(
      domain, params.diffusionCoefficient, params.sink,
      params.scallopStreamVelocity, params.holeStreamVelocity,
      builder.getTopLayer(), builder.getHeight(), builder.getHoleRadius());
  auto velField = psSmartPointer<psDefaultVelocityField<NumericType>>::New();
  auto etchSurfModel =
      psSmartPointer<SelectiveEtchingSurfaceModel<NumericType>>::New();
  auto depoSurfModel =
      psSmartPointer<RedepositionSurfaceModel<NumericType, D>>::New(
          cellSet, 1., builder.getHeight(), builder.getTopLayer());

  // run the selective etching process
  {
    auto etchModel = psSmartPointer<psProcessModel<NumericType, D>>::New();
    etchModel->setSurfaceModel(etchSurfModel);
    etchModel->setVelocityField(velField);
    etchModel->setAdvectionCallback(redepoModel);
    etchModel->setProcessName("SelectiveEtching");

    psProcess<NumericType, D> processEtch;
    processEtch.setDomain(domain);
    processEtch.setProcessModel(etchModel);
    processEtch.setProcessDuration(50.);

    processEtch.apply();
  }

  psWriteVisualizationMesh<NumericType, D>(domain, "FinalStack").apply();

  return 0;
}