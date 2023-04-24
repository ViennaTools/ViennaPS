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

  psLogger::setLogLevel(psLogLevel::TIMING);
  std::cout << "Log level " << psLogger::getLogLevel() << std::endl;

  // particles
  auto particle = std::make_unique<Particle<NumericType, D>>();

  // surface model
  auto surfModel = psSmartPointer<SurfaceModel<NumericType>>::New();

  // velocity field
  auto velField = psSmartPointer<VelocityField<NumericType>>::New();

  /* ------------- Geometry setup ------------ */
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

  auto plane = psSmartPointer<lsDomain<NumericType, D>>::New(
      bounds, boundaryCons, gridDelta);
  {
    NumericType normal[3] = {0.};
    NumericType origin[3] = {0.};
    normal[D - 1] = 1.;
    lsMakeGeometry<NumericType, D>(
        plane, lsSmartPointer<lsPlane<NumericType, D>>::New(origin, normal))
        .apply();
  }
  auto domain = psSmartPointer<psDomain<NumericType, D>>::New(plane);

  auto model = psSmartPointer<psProcessModel<NumericType, D>>::New();
  model->insertNextParticleType(particle);
  model->setSurfaceModel(surfModel);
  model->setVelocityField(velField);
  model->setProcessName("ExampleProcess");

  psProcess<NumericType, D> process;
  process.setDomain(domain);
  process.setProcessModel(model);
  process.setSourceDirection(D == 3 ? rayTraceDirection::POS_Z
                                    : rayTraceDirection::POS_Y);
  process.setProcessDuration(5);
  process.setMaxCoverageInitIterations(10);
  process.apply();

  domain->printSurface("ExampleProcess.vtp");

  return 0;
}