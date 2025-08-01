#include <geometries/psMakeHole.hpp>
#include <lsMakeGeometry.hpp>

#include <psProcess.hpp>
#include <psProcessModel.hpp>

#include "particles.hpp"
#include "surfaceModel.hpp"
#include "velocityField.hpp"

namespace ps = viennaps;
namespace ls = viennals;

int main() {
  using NumericType = double;
  constexpr int D = 3;

  ps::Logger::setLogLevel(ps::LogLevel::INFO);

  // particles
  auto particle = std::make_unique<Particle<NumericType, D>>(0.2, 10.);

  // surface model
  auto surfModel = ps::SmartPointer<SurfaceModel<NumericType>>::New();

  // velocity field
  auto velField = ps::SmartPointer<VelocityField<NumericType, D>>::New();

  // ------------- Geometry setup (ViennaLS) ------------
  auto domain = ps::SmartPointer<ps::Domain<NumericType, D>>::New();
  {
    NumericType extent = 8;
    NumericType gridDelta = 0.5;
    double bounds[2 * D] = {0};
    for (int i = 0; i < 2 * D; ++i)
      bounds[i] = i % 2 == 0 ? -extent : extent;
    ls::Domain<NumericType, D>::BoundaryType boundaryCons[D];
    for (int i = 0; i < D - 1; ++i)
      boundaryCons[i] =
          ls::Domain<NumericType, D>::BoundaryType::REFLECTIVE_BOUNDARY;
    boundaryCons[D - 1] =
        ls::Domain<NumericType, D>::BoundaryType::INFINITE_BOUNDARY;

    auto mask = ps::SmartPointer<ls::Domain<NumericType, D>>::New(
        bounds, boundaryCons, gridDelta);

    NumericType normal[3] = {0.};
    NumericType origin[3] = {0.};
    normal[D - 1] = -1.;
    ls::MakeGeometry<NumericType, D>(
        mask, ps::SmartPointer<ls::Plane<NumericType, D>>::New(origin, normal))
        .apply();

    auto maskCut = ps::SmartPointer<ls::Domain<NumericType, D>>::New(
        bounds, boundaryCons, gridDelta);
    normal[D - 1] = 1.;
    origin[D - 1] = 2.;
    ls::MakeGeometry<NumericType, D>(
        maskCut,
        ps::SmartPointer<ls::Plane<NumericType, D>>::New(origin, normal))
        .apply();

    ls::BooleanOperation<NumericType, D>(mask, maskCut,
                                         ls::BooleanOperationEnum::INTERSECT)
        .apply();

    origin[D - 1] = 0;
    normal[D - 1] = 1.;
    ls::MakeGeometry<NumericType, D>(
        maskCut, ps::SmartPointer<ls::Cylinder<NumericType, D>>::New(
                     origin, normal, 2. + gridDelta, 5.))
        .apply();

    ls::BooleanOperation<NumericType, D>(
        mask, maskCut, ls::BooleanOperationEnum::RELATIVE_COMPLEMENT)
        .apply();

    domain->insertNextLevelSetAsMaterial(mask, ps::Material::Mask);

    auto substrate = ps::SmartPointer<ls::Domain<NumericType, D>>::New(
        bounds, boundaryCons, gridDelta);

    ls::MakeGeometry<NumericType, D>(
        substrate,
        ps::SmartPointer<ls::Plane<NumericType, D>>::New(origin, normal))
        .apply();

    domain->insertNextLevelSetAsMaterial(substrate, ps::Material::Si);
  }

  domain->saveSurfaceMesh("initial.vtp");

  auto model = ps::SmartPointer<ps::ProcessModel<NumericType, D>>::New();
  model->insertNextParticleType(particle);
  model->setSurfaceModel(surfModel);
  model->setVelocityField(velField);
  model->setProcessName("ExampleProcess");

  ps::Process<NumericType, D> process;
  process.setDomain(domain);
  process.setProcessModel(model);
  process.setProcessDuration(5);
  process.apply();

  domain->saveSurfaceMesh("ExampleProcess.vtp");

  return 0;
}
