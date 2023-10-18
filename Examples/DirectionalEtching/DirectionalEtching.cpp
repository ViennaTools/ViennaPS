#include <lsMakeGeometry.hpp>

#include <psMakeTrench.hpp>
#include <psPointData.hpp>
#include <psProcess.hpp>
#include <psProcessModel.hpp>

#include "Particles.hpp"
#include "SurfaceModel.hpp"
#include "VelocityField.hpp"

int main() {
  using NumericType = double;
  constexpr int D = 2;

  omp_set_num_threads(20);

  psLogger::setLogLevel(psLogLevel::INFO);

  // particles
  auto particle = std::make_unique<Particle<NumericType, D>>(0.5, 100.);

  // surface model
  auto surfModel = psSmartPointer<SurfaceModel<NumericType>>::New();

  // velocity field
  auto velField = psSmartPointer<VelocityField<NumericType>>::New();

  /* ------------- Geometry setup (ViennaLS) ------------ */
  auto domain = psSmartPointer<psDomain<NumericType, D>>::New();
  NumericType gridDelta = 0.2;
  NumericType extent = 15.;
  NumericType width = 6.;
  NumericType depth = 4.;
  NumericType taper = 10.;
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
  process.setProcessDuration(25);
  process.setNumberOfRaysPerPoint(1000);
  process.apply();

  domain->printSurface("ExampleProcess.vtp");

  return 0;
}