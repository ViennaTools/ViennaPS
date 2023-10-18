#include <lsMakeGeometry.hpp>

#include <psMakeTrench.hpp>
#include <psPointData.hpp>
#include <psProcess.hpp>
#include <psProcessModel.hpp>

#include "Particle.hpp"
#include "SurfaceModel.hpp"

int main() {
  using NumericType = double;
  constexpr int D = 2;

  omp_set_num_threads(20);

  psLogger::setLogLevel(psLogLevel::INFO);

  // particles
  auto particle = std::make_unique<Particle<NumericType, D>>(0.7, 100.);

  // surface model
  auto surfModel = psSmartPointer<SurfaceModel<NumericType>>::New();

  // velocity field
  auto velField = psSmartPointer<VelocityField<NumericType, D>>::New();

  /* ------------- Geometry setup (ViennaLS) ------------ */
  auto domain = psSmartPointer<psDomain<NumericType, D>>::New();
  NumericType gridDelta = 0.01;
  NumericType extent = 2.;
  NumericType width = 0.3;
  NumericType depth = 0.4;
  NumericType taper = 35;
  psMakeTrench<NumericType, D>(domain, gridDelta, extent, extent, width, depth,
                               taper, 0., false, true, psMaterial::Si)
      .apply();

  domain->printSurface("initial.vtp");

  auto model = psSmartPointer<psProcessModel<NumericType, D>>::New();
  model->insertNextParticleType(particle);
  model->setSurfaceModel(surfModel);
  model->setVelocityField(velField);
  model->setProcessName("ExampleProcess");

  psProcess<NumericType, D> process;
  process.setDomain(domain);
  process.setProcessModel(model);
  process.setProcessDuration(2);
  process.setNumberOfRaysPerPoint(1000);
  process.apply();

  domain->printSurface("ExampleProcess.vtp");

  return 0;
}