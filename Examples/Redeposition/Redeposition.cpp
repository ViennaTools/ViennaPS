#include <csDenseCellSet.hpp>
#include <csTracing.hpp>

#include <psMakeStack.hpp>
#include <psProcess.hpp>
#include <psProcessModel.hpp>
#include <psWriteVisualizationMesh.hpp>

#include <StackRedeposition.hpp>

int main(int argc, char **argv) {
  using NumericType = double;

  omp_set_num_threads(12);
  // Attention:
  // This model/example currently only works in 2D mode
  constexpr int D = 2;
  const NumericType gridDelta = 0.2;

  const NumericType diffusionCoefficient = 10.; // diffusion cofficient
  const NumericType sink = 0.001;               // sink strength
  // convection velocity in the scallops towards the center
  const NumericType scallopStreamVelocity = 10.;
  // convection velocity in the center towards the sink on the top
  const NumericType holeStreamVelocity = 10.;
  // number of stacked layers
  int numLayers = 26;

  auto domain = psSmartPointer<psDomain<NumericType, D>>::New();
  psMakeStack<NumericType, D> builder(domain, numLayers, gridDelta);
  builder.apply();
  // copy top layer for deposition
  auto depoLayer = psSmartPointer<lsDomain<NumericType, D>>::New(
      domain->getLevelSets()->back());
  domain->insertNextLevelSet(depoLayer);
  domain->generateCellSet(2., true /* true means cell set above surface */);
  auto &cellSet = domain->getCellSet();
  cellSet->writeVTU("initial.vtu");
  // we need neighborhood information for solving the
  // convection-diffusion equation on the cell set
  cellSet->buildNeighborhood();

  // The redeposition model captures byproducts from the selective etching
  // process in the cell set. The byproducts are then distributed by solving a
  // convection-diffusion equation on the cell set.
  auto redepoModel = psSmartPointer<RedepositionDynamics<NumericType, D>>::New(
      domain, diffusionCoefficient, sink, scallopStreamVelocity,
      holeStreamVelocity, builder.getTopLayer(), builder.getHeight(),
      builder.getHoleRadius());
  auto velField = psSmartPointer<psDefaultVelocityField<NumericType>>::New();
  auto etchSurfModel =
      psSmartPointer<SelectiveEtchingSurfaceModel<NumericType>>::New();
  auto depoSurfModel =
      psSmartPointer<RedepositionSurfaceModel<NumericType, D>>::New(
          cellSet, 1., builder.getHeight(), builder.getTopLayer());

  std::cout << builder.getTopLayer() << std::endl;

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

  // run the redeposition based on the captured byproducts
  {
    auto depoModel = psSmartPointer<psProcessModel<NumericType, D>>::New();
    depoModel->setSurfaceModel(depoSurfModel);
    depoModel->setVelocityField(velField);
    depoModel->setProcessName("Redeposition");

    psProcess<NumericType, D> processRedepo;
    processRedepo.setDomain(domain);
    processRedepo.setProcessModel(depoModel);
    processRedepo.setProcessDuration(0.2);

    processRedepo.apply();
    cellSet->updateMaterials();
    cellSet->writeVTU("RedepositionCellSet.vtu");
  }

  psWriteVisualizationMesh<NumericType, D>(domain, "FinalStack").apply();

  return 0;
}