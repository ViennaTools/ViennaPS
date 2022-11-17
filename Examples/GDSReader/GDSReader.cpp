#include <psDomain.hpp>
#include <psGDSReader.hpp>
#include <psPointValuesToLevelSet.hpp>
#include <psProcess.hpp>
#include <psToDiskMesh.hpp>
#include <psVTKWriter.hpp>

#include "DirectionalEtch.hpp"
#include "IsoDeposition.hpp"

template <class NumericType, int D>
void printSurface(psSmartPointer<psDomain<NumericType, D>> domain,
                  std::string name) {
  auto translator =
      psSmartPointer<std::unordered_map<unsigned long, unsigned long>>::New();
  auto mesh = psSmartPointer<lsMesh<NumericType>>::New();
  psToDiskMesh<NumericType, D>(domain, mesh, translator).apply();
  auto matIds = mesh->getCellData().getScalarData("MaterialIds");
  psPointValuesToLevelSet<NumericType, D>(domain->getLevelSets()->back(),
                                          translator, matIds, "Material")
      .apply();
  domain->printSurface(name);
}

int main(int argc, char **argv) {
  using NumericType = double;
  constexpr int D = 3;

  // read GDS mask file
  const NumericType gridDelta = 0.01;
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
  NumericType *bounds = mask->getBounds();
  bounds[0] = -0.04;
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
        bounds, boundaryCons, gridDelta);
    lsMakeGeometry<NumericType, D>(
        plane, psSmartPointer<lsPlane<NumericType, D>>::New(origin, normal))
        .apply();

    geometry->insertNextLevelSet(plane);
  }

  // directional etching
  {
    auto model = psSmartPointer<psProcessModel<NumericType, D>>::New();
    auto surfModel =
        psSmartPointer<DirectionalEtchSurfaceModel<NumericType>>::New();
    auto velField =
        psSmartPointer<DirectionalEtchVelocityField<NumericType>>::New();
    model->setProcessName("directionalEtch");
    model->setSurfaceModel(surfModel);
    model->setVelocityField(velField);

    psProcess<NumericType, D> process;
    process.setDomain(geometry);
    process.setProcessDuration(4);
    process.setProcessModel(model);
    process.apply();

    printSurface(geometry, "dirEtch.vtp");
  }

  // remove mask
  {
    geometry->getLevelSets()->erase(geometry->getLevelSets()->begin());
    auto substrate = geometry->getLevelSets()->back();
    NumericType origin[D] = {0., 0., 0.};
    NumericType normal[D] = {0., 0., 1.};
    auto plane = psSmartPointer<lsDomain<NumericType, D>>::New(
        bounds, boundaryCons, gridDelta);
    lsMakeGeometry<NumericType, D>(
        plane, psSmartPointer<lsPlane<NumericType, D>>::New(origin, normal))
        .apply();
    lsBooleanOperation<NumericType, D>(substrate, plane,
                                       lsBooleanOperationEnum::INTERSECT)
        .apply();

    printSurface(geometry, "maskRemoved.vtp");
  }

  // add STI
  {
    NumericType origin[D] = {0., 0., -0.15};
    NumericType normal[D] = {0., 0., 1.};
    auto plane = psSmartPointer<lsDomain<NumericType, D>>::New(
        bounds, boundaryCons, gridDelta);
    lsMakeGeometry<NumericType, D>(
        plane, psSmartPointer<lsPlane<NumericType, D>>::New(origin, normal))
        .apply();

    geometry->insertNextLevelSet(plane);

    printSurface(geometry, "STI.vtp");
  }

  // add STI
  {
    auto fins = mask->layerToLevelSet(1 /*layer*/, -0.15 /*base z position*/,
                                      0.45 /*height*/);
    geometry->insertNextLevelSet(fins);
    printSurface(geometry, "Add.vtp");
  }

  // isotropic deposition
  {
    auto depoLayer = psSmartPointer<lsDomain<NumericType, D>>::New(
        geometry->getLevelSets()->back());
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

    geometry->printSurface("isoDepo.vtp");
  }

  return 0;
}