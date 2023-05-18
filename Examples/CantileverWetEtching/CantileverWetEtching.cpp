#include <psDomain.hpp>
#include <psGDSReader.hpp>
#include <psPlanarize.hpp>
#include <psProcess.hpp>

#include <WetEtching.hpp>

int main(int argc, char **argv) {
  using NumericType = double;
  constexpr int D = 3;

  // set number threads to be used
  omp_set_num_threads(12);

  std::string maskFileName = "cantilever_mask.gds";

  const int minutes = 120 / 5;   // total etch time (2 hours)
  const NumericType x_add = 50.; // add space to domain boundary
  const NumericType y_add = 50.;
  const NumericType gridDelta = 5.; // um

  /* -------- read GDS mask file --------- */
  typename lsDomain<NumericType, D>::BoundaryType boundaryCons[D];
  for (int i = 0; i < D - 1; i++)
    boundaryCons[i] = lsDomain<NumericType, D>::BoundaryType::
        REFLECTIVE_BOUNDARY; // boundary conditions in x and y direction
  boundaryCons[D - 1] = lsDomain<NumericType, D>::BoundaryType::
      INFINITE_BOUNDARY; // open boundary in z direction
  auto gds_mask = psSmartPointer<psGDSGeometry<NumericType, D>>::New(gridDelta);
  gds_mask->setBoundaryConditions(boundaryCons);
  gds_mask->setBoundaryPadding(x_add, y_add);
  psGDSReader<NumericType, D>(gds_mask, maskFileName).apply(); // read GDS file

  auto mask = gds_mask->layerToLevelSet(
      1 /*layer in GDS file*/, 0 /*base z position*/,
      4 * gridDelta /*mask height*/, true /*invert mask*/);

  /* -------- create plane substrate --------- */
  NumericType origin[D] = {0., 0., 0.};   // surface origin
  NumericType normal[D] = {0., 0., 1.};   // surface normal
  double *bounds = gds_mask->getBounds(); // extent of GDS mask
  auto plane = psSmartPointer<lsDomain<NumericType, D>>::New(
      bounds, boundaryCons, gridDelta);
  lsMakeGeometry<NumericType, D>(
      plane, psSmartPointer<lsPlane<NumericType, D>>::New(origin, normal))
      .apply();

  /* -------- set up geometry --------- */
  auto geometry = psSmartPointer<psDomain<NumericType, D>>::New();
  geometry->insertNextLevelSet(mask);
  geometry->insertNextLevelSet(plane);
  geometry->printSurface("InitialGeometry.vtp");

  /* -------- wet etch process --------- */
  WetEtching<NumericType, D> wetEtchModel;

  psProcess<NumericType, D> process;
  process.setDomain(geometry);
  process.setProcessModel(wetEtchModel.getProcessModel());
  process.setProcessDuration(5. * 60.); // 5 minutes of etching
  process.setPrintIntermediate(false);
  process.setIntegrationScheme(
      lsIntegrationSchemeEnum::STENCIL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER);

  for (int n = 0; n < minutes; n++) {
    process.apply(); // run process
    geometry->printSurface("WetEtchingSurface_" + std::to_string(n) + ".vtp");
  }

  geometry->printSurface("FinalGeometry.vtp");
}