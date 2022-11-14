#include <psDomain.hpp>
#include <psGDSReader.hpp>
#include <psPointValuesToLevelSet.hpp>
#include <psProcess.hpp>
#include <psToDiskMesh.hpp>
#include <psVTKWriter.hpp>

#include "DirectionalEtch.hpp"
#include "Epitaxy.hpp"
#include "IsoDeposition.hpp"

template <class NumericType, int D>
void printSurface(psSmartPointer<psDomain<NumericType, D>> domain,
                  std::string name) {
  auto translator =
      psSmartPointer<std::unordered_map<unsigned long, unsigned long>>::New();
  auto mesh = psSmartPointer<lsMesh<NumericType>>::New();
  psToDiskMesh<NumericType, D>(domain, mesh, translator).apply();
  auto matIds = mesh->getCellData().getScalarData("MaterialIds");
  psPointValuesToLevelSet<NumericType, D>(domain->getSurfaceLevelSet(),
                                          translator, matIds, "Material")
      .apply();
  domain->printSurface(name);
}

int main(int argc, char **argv) {
  using NumericType = double;
  constexpr int D = 3;

  // read GDS mask file
  const NumericType gridDelta = 0.005;
  typename lsDomain<NumericType, D>::BoundaryType boundaryCons[D];
  for (int i = 0; i < D - 1; i++)
    boundaryCons[i] =
        lsDomain<NumericType, D>::BoundaryType::REFLECTIVE_BOUNDARY;
  boundaryCons[D - 1] =
      lsDomain<NumericType, D>::BoundaryType::INFINITE_BOUNDARY;
  auto mask = psSmartPointer<psGDSGeometry<NumericType, D>>::New(gridDelta);
  mask->setBoundaryConditions(boundaryCons);
  psGDSReader<NumericType, D>(mask, "SRAM_mask.gds").apply();

  // geometry setup
  auto bounds = mask->getBounds();
  auto geometry = psSmartPointer<psDomain<NumericType, D>>::New();

  // fin patterning
  {
    auto fins = mask->layerToLevelSet(0 /*layer*/, 0 /*base z position*/,
                                      0.1 /*height*/);
    geometry->insertNextLevelSet(fins);

    // substrate plane
    NumericType origin[D] = {0., 0., 0.};
    NumericType normal[D] = {0., 0., 1.};
    auto plane = psSmartPointer<lsDomain<NumericType, D>>::New(
        bounds.data(), boundaryCons, gridDelta);
    lsMakeGeometry<NumericType, D>(
        plane, psSmartPointer<lsPlane<NumericType, D>>::New(origin, normal))
        .apply();

    geometry->insertNextLevelSet(plane);
    printSurface(geometry, "step_0.vtp");
  }

  // directional etching
  {
    auto model = psSmartPointer<psProcessModel<NumericType, D>>::New();
    auto surfModel =
        psSmartPointer<DirectionalEtchSurfaceModel<NumericType>>::New();
    auto velField =
        psSmartPointer<DirectionalEtchVelocityField<NumericType>>::New(1);
    model->setProcessName("directionalEtch");
    model->setSurfaceModel(surfModel);
    model->setVelocityField(velField);

    psProcess<NumericType, D> process;
    process.setDomain(geometry);
    process.setProcessDuration(4);
    process.setProcessModel(model);
    process.apply();

    printSurface(geometry, "step_1.vtp");
  }

  // remove mask
  {
    geometry->removeLevelSet(0);
    printSurface(geometry, "step_2.vtp");
  }

  // add STI
  {
    NumericType origin[D] = {0., 0., -0.15};
    NumericType normal[D] = {0., 0., 1.};
    auto plane = psSmartPointer<lsDomain<NumericType, D>>::New(
        bounds.data(), boundaryCons, gridDelta);
    lsMakeGeometry<NumericType, D>(
        plane, psSmartPointer<lsPlane<NumericType, D>>::New(origin, normal))
        .apply();

    geometry->insertNextLevelSet(plane);

    printSurface(geometry, "step_3.vtp");
  }

  // add STI
  {
    auto fins = mask->layerToLevelSet(1 /*layer*/, -0.15 /*base z position*/,
                                      0.46 /*height*/);
    geometry->insertNextLevelSet(fins);
    printSurface(geometry, "step_4.vtp");
  }

  // isotropic deposition
  {
    auto depoLayer = psSmartPointer<lsDomain<NumericType, D>>::New(
        geometry->getSurfaceLevelSet());
    geometry->insertNextLevelSet(depoLayer);
    auto model = psSmartPointer<psProcessModel<NumericType, D>>::New();
    auto surfModel =
        psSmartPointer<IsoDepositionSurfaceModel<NumericType>>::New();
    auto velField =
        psSmartPointer<IsoDepositionVelocityField<NumericType>>::New();
    model->setProcessName("isoDeposition");
    model->setSurfaceModel(surfModel);
    model->setVelocityField(velField);

    psProcess<NumericType, D> process;
    process.setDomain(geometry);
    process.setProcessDuration(1.5);
    process.setProcessModel(model);
    process.apply();

    std::cout << geometry->getLevelSets()->size() << std::endl;
    printSurface(geometry, "step_5.vtp");
  }

  return 0;

  // dummy gate
  {
    auto gate = mask->layerToLevelSet(2 /*layer*/, -0.15 /*base z position*/,
                                      0.5 /*height*/);
    geometry->insertNextLevelSet(gate);
    printSurface(geometry, "step_6.vtp");
  }

  // PMOS spacer etching
  {
    auto model = psSmartPointer<psProcessModel<NumericType, D>>::New();
    auto surfModel =
        psSmartPointer<DirectionalEtchSurfaceModel<NumericType>>::New();
    auto velField =
        psSmartPointer<DirectionalEtchVelocityField<NumericType>>::New(3);
    model->setProcessName("directionalEtch");
    model->setSurfaceModel(surfModel);
    model->setVelocityField(velField);

    psProcess<NumericType, D> process;
    process.setDomain(geometry);
    process.setProcessDuration(1);
    process.setProcessModel(model);
    process.apply();

    printSurface(geometry, "step_7.vtp");
  }

  // fin recess etching
  {
    auto model = psSmartPointer<psProcessModel<NumericType, D>>::New();
    auto surfModel =
        psSmartPointer<DirectionalEtchSurfaceModel<NumericType>>::New();
    auto velField =
        psSmartPointer<DirectionalEtchVelocityField<NumericType>>::New(0);
    model->setProcessName("directionalEtch");
    model->setSurfaceModel(surfModel);
    model->setVelocityField(velField);

    psProcess<NumericType, D> process;
    process.setDomain(geometry);
    process.setProcessDuration(0.7);
    process.setProcessModel(model);
    process.apply();

    printSurface(geometry, "step_8.vtp");
  }

  // epitaxy
  {
    auto depoLayer = psSmartPointer<lsDomain<NumericType, D>>::New(
        geometry->getSurfaceLevelSet());
    geometry->insertNextLevelSet(depoLayer);
    auto model = psSmartPointer<psProcessModel<NumericType, D>>::New();
    auto surfModel = psSmartPointer<EpitaxySurfaceModel<NumericType>>::New();
    auto velField =
        psSmartPointer<EpitaxyVelocityField<NumericType>>::New(0, 5);
    model->setProcessName("epitaxy");
    model->setSurfaceModel(surfModel);
    model->setVelocityField(velField);

    psProcess<NumericType, D> process;
    process.setDomain(geometry);
    process.setProcessDuration(.3);
    process.setProcessModel(model);
    process.apply();

    printSurface(geometry, "step_9.vtp");
  }

  return 0;
}