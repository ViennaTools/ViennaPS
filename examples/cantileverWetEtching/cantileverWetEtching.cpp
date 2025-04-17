#include <geometries/psMakePlane.hpp>
#include <models/psAnisotropicProcess.hpp>
#include <psDomain.hpp>
#include <psGDSReader.hpp>
#include <psPlanarize.hpp>
#include <psProcess.hpp>

namespace ps = viennaps;

int main(int argc, char **argv) {
  using NumericType = double;
  constexpr int D = 3;

  // set number threads to be used
  omp_set_num_threads(16);

  const std::string maskFileName = "cantilever_mask.gds";

  // crystal surface direction
  const ps::Vec3D<NumericType> direction100{0.707106781187, 0.707106781187, 0.};
  const ps::Vec3D<NumericType> direction010{-0.707106781187, 0.707106781187,
                                            0.};
  // etch rates for crystal directions in um / s
  // 30 % KOH at 70°C
  // https://doi.org/10.1016/S0924-4247(97)01658-0
  const NumericType r100 = 0.797 / 60.;
  const NumericType r110 = 1.455 / 60.;
  const NumericType r111 = 0.005 / 60.;
  const NumericType r311 = 1.436 / 60.;
  const int minutes = 120 / 5; // total etch time (2 hours)

  const NumericType x_add = 50.; // add space to domain boundary
  const NumericType y_add = 50.;
  const NumericType gridDelta = 5.; // um

  // Read GDS file and convert to level set
  ps::BoundaryType boundaryCons[D];
  for (int i = 0; i < D - 1; i++)
    boundaryCons[i] =
        ps::BoundaryType::REFLECTIVE_BOUNDARY; // boundary conditions in x and y
                                               // direction
  boundaryCons[D - 1] =
      ps::BoundaryType::INFINITE_BOUNDARY; // open boundary in z direction
  auto gds_mask =
      ps::SmartPointer<ps::GDSGeometry<NumericType, D>>::New(gridDelta);
  gds_mask->setBoundaryConditions(boundaryCons);
  gds_mask->setBoundaryPadding(x_add, y_add);
  ps::GDSReader<NumericType, D>(gds_mask, maskFileName)
      .apply(); // read GDS file

  auto mask = gds_mask->layerToLevelSet(
      1 /*layer in GDS file*/, 0 /*base z position*/,
      4 * gridDelta /*mask height*/, true /*invert mask*/, false /*blur*/);

  // Set up domain
  auto geometry = ps::SmartPointer<ps::Domain<NumericType, D>>::New();
  geometry->insertNextLevelSetAsMaterial(mask, ps::Material::Mask);

  // Create plane substrate under mask
  ps::MakePlane<NumericType, D>(geometry, 0., ps::Material::Si, true).apply();

  geometry->saveSurfaceMesh("initialGeometry.vtp");

  // Anisotropic wet etching process model
  auto model = ps::SmartPointer<ps::AnisotropicProcess<NumericType, D>>::New(
      direction100, direction010, r100, r110, r111, r311,
      std::vector<std::pair<ps::Material, NumericType>>{
          {ps::Material::Si, -1.}});

  ps::Process<NumericType, D> process;
  process.setDomain(geometry);
  process.setProcessModel(model);
  process.setProcessDuration(5. * 60.); // 5 minutes of etching
  process.setIntegrationScheme(
      viennals::IntegrationSchemeEnum::STENCIL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER);

  for (int n = 0; n < minutes; n++) {
    process.apply(); // run process
    geometry->saveSurfaceMesh("wetEtchingSurface_" + std::to_string(n) +
                              ".vtp");
  }

  geometry->saveSurfaceMesh("finalGeometry.vtp");
}
