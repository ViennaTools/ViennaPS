#include <lsMakeGeometry.hpp>
#include <lsToSurfaceMesh.hpp>
#include <lsVTKWriter.hpp>
#include <psPointData.hpp>
#include <psProcess.hpp>
#include <psProcessModel.hpp>
#include <psSmartPointer.hpp>
#include <rayTracingData.hpp>

#include "particles.hpp"
#include "surfaceModel.hpp"
#include "velocityField.hpp"

class myCellType : public cellBase {
  using cellBase::cellBase;
};

template <typename T, int D>
void printLS(lsSmartPointer<lsDomain<T, D>> dom, std::string name) {
  auto mesh = lsSmartPointer<lsMesh<T>>::New();
  lsToSurfaceMesh<T, D>(dom, mesh).apply();
  lsVTKWriter<T>(mesh, name).apply();
}

int main() {
  omp_set_num_threads(8);
  using NumericType = double;
  constexpr int D = 3;

  // particles
  auto particle = std::make_unique<Particle<NumericType, D>>();

  // surface model
  auto surfModel = psSmartPointer<surfaceModel<NumericType>>::New();

  // velocity field
  auto velField = psSmartPointer<velocityField<NumericType>>::New();

  /* ------------- Geometry setup ------------ */
  // domain
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
    NumericType normal[3] = {0., 0., 1.};
    NumericType origin[3] = {0.};
    lsMakeGeometry<NumericType, D>(
        plane, lsSmartPointer<lsPlane<NumericType, D>>::New(origin, normal))
        .apply();
  }
  auto domain =
      psSmartPointer<psDomain<myCellType, NumericType, D>>::New(plane);

  auto model = psSmartPointer<psProcessModel<NumericType>>::New();
  model->insertNextParticleType(particle);
  model->setSurfaceModel(surfModel);
  model->setVelocityField(velField);

  psProcess<myCellType, NumericType, D> process;
  process.setDomain(domain);
  process.setProcessModel(model);
  process.setSourceDirection(rayTraceDirection::POS_Z);
  process.setProcessDuration(100);
  process.apply();

  auto mesh = lsSmartPointer<lsMesh<NumericType>>::New();
  lsToSurfaceMesh<NumericType, D>(plane, mesh).apply();

  lsVTKWriter<NumericType>(mesh, "example.vtp").apply();

  return 0;
}