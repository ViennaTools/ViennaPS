// #define VIENNALS_USE_VTK

#include <lsDomain.hpp>
#include <lsWriteVisualizationMesh.hpp>
#include <psProcess.hpp>

#include "geometryFactory.hpp"
#include "particles.hpp"
#include "surfaceModel.hpp"
#include "velocityField.hpp"

class myCellType : public cellBase {
  using cellBase::cellBase;
};

int main() {
  omp_set_num_threads(8);
  using NumericType = double;
  constexpr int D = 2;

  /* ------------- Geometry setup ------------ */
  // domain
  NumericType extent = 10;
  NumericType gridDelta = 0.1;
  double bounds[2 * D] = {0};
  for (int i = 0; i < 2 * D; ++i)
    bounds[i] = i % 2 == 0 ? -extent : extent;
  lsDomain<NumericType, D>::BoundaryType boundaryCons[D];
  for (int i = 0; i < D - 1; ++i)
    boundaryCons[i] =
        lsDomain<NumericType, D>::BoundaryType::REFLECTIVE_BOUNDARY;
  boundaryCons[D - 1] =
      lsDomain<NumericType, D>::BoundaryType::INFINITE_BOUNDARY;

  // Create via mask on top
  auto mask = psSmartPointer<lsDomain<NumericType, D>>::New(
      bounds, boundaryCons, gridDelta);
  {
    std::array<NumericType, 3> maskOrigin = {0.};
    MakeMask<NumericType, D> makeMask(mask);
    makeMask.setMaskOrigin(maskOrigin);
    makeMask.setMaskRadius(4);
    makeMask.apply();
  }
  auto domain = psSmartPointer<psDomain<myCellType, NumericType, D>>::New(mask);

  // Material layer
  auto substrate = MakeLayer<NumericType, D>(mask).apply();
  auto polymer = psSmartPointer<lsDomain<NumericType, D>>::New(substrate);
  domain->insertNextLevelSet(substrate);
  domain->insertNextLevelSet(polymer);

  // visualization
  lsWriteVisualizationMesh<NumericType, D> volumeMesh;
  volumeMesh.insertNextLevelSet(mask);
  volumeMesh.insertNextLevelSet(substrate);
  volumeMesh.insertNextLevelSet(polymer);

  auto veloField = psSmartPointer<velocityField<NumericType>>::New();

  // surface models
  auto clearModel = psSmartPointer<Clear<NumericType>>::New();
  auto oxidizeModel = psSmartPointer<Oxidize<NumericType>>::New();
  auto removeModel = psSmartPointer<Remove<NumericType>>::New();
  auto etchModel = psSmartPointer<Etch<NumericType>>::New();

  // particles
  auto ionParticle = std::make_unique<Ion<NumericType>>();
  auto etchantParticle = std::make_unique<Etchant<NumericType, D>>();

  // process models
  auto clearProcessModel = psSmartPointer<psProcessModel<NumericType>>::New();
  clearProcessModel->setSurfaceModel(clearModel);
  clearProcessModel->setVelocityField(veloField);
  clearProcessModel->setProcessName("clear");

  auto oxidizeProcessModel = psSmartPointer<psProcessModel<NumericType>>::New();
  oxidizeProcessModel->setSurfaceModel(oxidizeModel);
  oxidizeProcessModel->setVelocityField(veloField);
  oxidizeProcessModel->setProcessName("oxidize");

  auto removeProcessModel = psSmartPointer<psProcessModel<NumericType>>::New();
  removeProcessModel->setSurfaceModel(removeModel);
  removeProcessModel->setVelocityField(veloField);
  removeProcessModel->insertNextParticleType(ionParticle);
  removeProcessModel->setProcessName("remove");

  auto etchProcessModel = psSmartPointer<psProcessModel<NumericType>>::New();
  etchProcessModel->setSurfaceModel(etchModel);
  etchProcessModel->setVelocityField(veloField);
  // neglect ion enhanced etching and ion sputtering in first approximation
  // etchProcessModel->insertNextParticleType(ionParticle);
  etchProcessModel->insertNextParticleType(etchantParticle);
  etchProcessModel->setProcessName("etch");

  // processes
  psProcess<myCellType, NumericType, D> clearProcess;
  clearProcess.setDomain(domain);
  clearProcess.setProcessModel(clearProcessModel);

  psProcess<myCellType, NumericType, D> oxidizeProcess;
  oxidizeProcess.setDomain(domain);
  oxidizeProcess.setProcessModel(oxidizeProcessModel);

  psProcess<myCellType, NumericType, D> removeProcess;
  removeProcess.setDomain(domain);
  removeProcess.setSourceDirection(rayTraceDirection::POS_Y);
  removeProcess.setProcessModel(removeProcessModel);

  psProcess<myCellType, NumericType, D> etchProcess;
  etchProcess.setDomain(domain);
  etchProcess.setSourceDirection(rayTraceDirection::POS_Y);
  etchProcess.setProcessModel(etchProcessModel);
  etchProcess.setNumberOfRaysPerPoint(2000);

  size_t counter = 0;
  volumeMesh.setFileName("Mesh_" + std::to_string(counter++));
  volumeMesh.apply();

  for (int i = 0; i < 3; i++) {
    std::cout << "Cycle " << i << std::endl;

    // oxidize
    oxidizeProcess.setProcessDuration(0.5);
    oxidizeProcess.apply();
    volumeMesh.setFileName("Mesh_" + std::to_string(counter++));
    volumeMesh.apply();

    // remove
    removeProcess.setProcessDuration(2);
    removeProcess.apply();
    volumeMesh.setFileName("Mesh_" + std::to_string(counter++));
    volumeMesh.apply();

    // etch
    etchProcess.setProcessDuration(15. + NumericType(i));
    etchProcess.apply();
    volumeMesh.setFileName("Mesh_" + std::to_string(counter++));
    volumeMesh.apply();

    // clear
    clearProcess.setProcessDuration(2);
    clearProcess.apply();
    volumeMesh.setFileName("Mesh_" + std::to_string(counter++));
    volumeMesh.apply();
  }

  return 0;
}