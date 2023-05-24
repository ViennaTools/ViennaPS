#include <lsMakeGeometry.hpp>

#include <Geometries/psMakeHole.hpp>
#include <psPointData.hpp>
#include <psProcess.hpp>
#include <psProcessModel.hpp>
#include <psSmartPointer.hpp>
#include <psToSurfaceMesh.hpp>
#include <psVTKWriter.hpp>

#include "Particles.hpp"
#include "SurfaceModel.hpp"
#include "VelocityField.hpp"

int main() {
  using NumericType = double;
  constexpr int D = 3;

  psLogger::setLogLevel(psLogLevel::INFO);

  // particles
  auto particle = std::make_unique<Particle<NumericType, D>>(0.2, 10.);

  // surface model
  auto surfModel = psSmartPointer<SurfaceModel<NumericType>>::New();

  // velocity field
  auto velField = psSmartPointer<VelocityField<NumericType>>::New();

  /* ------------- Geometry setup (ViennaLS) ------------ */
  auto domain = psSmartPointer<psDomain<NumericType, D>>::New();
  {
    NumericType extent = 8;
    NumericType gridDelta = 0.5;
    double bounds[2 * D] = {0};
    for (int i = 0; i < 2 * D; ++i)
      bounds[i] = i % 2 == 0 ? -extent : extent;
    lsDomain<NumericType, D>::BoundaryType boundaryCons[D];
    for (int i = 0; i < D - 1; ++i)
      boundaryCons[i] =
          lsDomain<NumericType, D>::BoundaryType::REFLECTIVE_BOUNDARY;
    boundaryCons[D - 1] =
        lsDomain<NumericType, D>::BoundaryType::INFINITE_BOUNDARY;

    auto mask = psSmartPointer<lsDomain<NumericType, D>>::New(
        bounds, boundaryCons, gridDelta);

    NumericType normal[3] = {0.};
    NumericType origin[3] = {0.};
    normal[D - 1] = -1.;
    lsMakeGeometry<NumericType, D>(
        mask, lsSmartPointer<lsPlane<NumericType, D>>::New(origin, normal))
        .apply();

    auto maskCut = psSmartPointer<lsDomain<NumericType, D>>::New(
        bounds, boundaryCons, gridDelta);
    normal[D - 1] = 1.;
    origin[D - 1] = 2.;
    lsMakeGeometry<NumericType, D>(
        maskCut, lsSmartPointer<lsPlane<NumericType, D>>::New(origin, normal))
        .apply();

    lsBooleanOperation<NumericType, D>(mask, maskCut,
                                       lsBooleanOperationEnum::INTERSECT)
        .apply();

    origin[D - 1] = 0;
    normal[D - 1] = 1.;
    lsMakeGeometry<NumericType, D>(
        maskCut, lsSmartPointer<lsCylinder<NumericType, D>>::New(
                     origin, normal, 2. + gridDelta, 5.))
        .apply();

    lsBooleanOperation<NumericType, D>(
        mask, maskCut, lsBooleanOperationEnum::RELATIVE_COMPLEMENT)
        .apply();

    domain->insertNextLevelSetAsMaterial(mask, psMaterial::Mask);

    auto substrate = psSmartPointer<lsDomain<NumericType, D>>::New(
        bounds, boundaryCons, gridDelta);

    lsMakeGeometry<NumericType, D>(
        substrate, lsSmartPointer<lsPlane<NumericType, D>>::New(origin, normal))
        .apply();

    domain->insertNextLevelSetAsMaterial(substrate, psMaterial::Si);
  }

  domain->printSurface("initial.vtp");

  auto model = psSmartPointer<psProcessModel<NumericType, D>>::New();
  model->insertNextParticleType(particle);
  model->setSurfaceModel(surfModel);
  model->setVelocityField(velField);
  model->setProcessName("ExampleProcess");

  psProcess<NumericType, D> process;
  process.setDomain(domain);
  process.setProcessModel(model);
  process.setProcessDuration(5);
  process.apply();

  domain->printSurface("ExampleProcess.vtp");

  return 0;
}