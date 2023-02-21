#include <csDenseCellSet.hpp>
#include <csTracing.hpp>

#include <psProcess.hpp>
#include <psProcessModel.hpp>
#include <psWriteVisualizationMesh.hpp>

#include <StackRedeposition.hpp>

#include "MakeStack.hpp"

int main(int argc, char **argv) {
  using NumericType = double;

  omp_set_num_threads(12);
  constexpr int D = 2;
  const NumericType gridDelta = 0.2;

  const NumericType diffusionCoefficient = 10.;
  const NumericType sink = 0.001;
  const NumericType scallopStreamVelocity = 10.;
  const NumericType holeStreamVelocity = 10.;
  int etchSteps = 50;
  int numLayers = 25;

  MakeStack<NumericType, D> builder(numLayers, gridDelta);
  auto domain = builder.makeStack();
  domain->generateCellSet(2., true /* cell set above surface */);
  auto &cellSet = domain->getCellSet();
  cellSet->writeVTU("initial.vtu");
  cellSet->buildNeighborhood();

  auto redepoModel = psSmartPointer<RedepositionDynamics<NumericType, D>>::New(
      domain, diffusionCoefficient, sink, scallopStreamVelocity,
      holeStreamVelocity, builder.getTopLayer() + 1, builder.getHeight(),
      builder.getHoleRadius());

  auto velField = psSmartPointer<psDefaultVelocityField<NumericType>>::New();
  auto etchSurfModel =
      psSmartPointer<SelectiveEtchingSurfaceModel<NumericType>>::New();
  auto depoSurfModel =
      psSmartPointer<RedepositionSurfaceModel<NumericType, D>>::New(
          cellSet, builder.getHeight(), builder.getTopLayer() + 1);

  auto etchModel = psSmartPointer<psProcessModel<NumericType, D>>::New();
  etchModel->setSurfaceModel(etchSurfModel);
  etchModel->setVelocityField(velField);
  etchModel->setAdvectionCallback(redepoModel);
  etchModel->setProcessName("SelectiveEtching");

  auto depoModel = psSmartPointer<psProcessModel<NumericType, D>>::New();
  depoModel->setSurfaceModel(depoSurfModel);
  depoModel->setVelocityField(velField);
  depoModel->setProcessName("Redeposition");

  psProcess<NumericType, D> processEtch;
  processEtch.setDomain(domain);
  processEtch.setProcessModel(etchModel);
  processEtch.setProcessDuration(etchSteps * 1.);

  psProcess<NumericType, D> processRedepo;
  processRedepo.setDomain(domain);
  processRedepo.setProcessModel(depoModel);
  processRedepo.setProcessDuration(0.2);

  processEtch.apply();
  domain->printSurface("EtchedSurface.vtp");

  processRedepo.apply();
  cellSet->updateMaterials();
  cellSet->writeVTU("RedepositionCellSet.vtu");

  psWriteVisualizationMesh<NumericType, D>(domain, "FinalStack").apply();

  return 0;
}