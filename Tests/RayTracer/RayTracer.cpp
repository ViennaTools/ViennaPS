#include <iostream>
#include <lsAdvect.hpp>
#include <lsBooleanOperation.hpp>
#include <lsDomain.hpp>
#include <lsMakeGeometry.hpp>
#include <lsToMesh.hpp>
#include <lsToSurfaceMesh.hpp>
#include <lsVTKWriter.hpp>
#include <rtTrace.hpp>

#include "particle.hpp"

using NumericType = float;
using reflection = rti::reflection::diffuse<NumericType>;

class velocityField : public lsVelocityField<NumericType> {
public:
  velocityField(std::vector<NumericType> &&_mcestimates) {
    mcestimates = _mcestimates;
  }

  NumericType getScalarVelocity(
      const std::array<NumericType, 3> & /*coordinate*/, int /*material*/,
      const std::array<NumericType, 3>
          & /*normalVector = hrleVectorType<NumericType, 3>(0.)*/) {
    return mcestimates[counter++];
  }

private:
  unsigned counter = 0;
  std::vector<NumericType> mcestimates;
};

int main() {
  constexpr int D = 3;

  NumericType eps = 1e-4;
  NumericType extent = 30;
  NumericType gridDelta = 0.5;
  double bounds[2 * D] = {-extent, extent, -extent, extent, -extent, extent};
  lsDomain<NumericType, D>::BoundaryType boundaryCons[D];
  boundaryCons[0] = lsDomain<NumericType, D>::BoundaryType::REFLECTIVE_BOUNDARY;
  boundaryCons[1] = lsDomain<NumericType, D>::BoundaryType::REFLECTIVE_BOUNDARY;
  boundaryCons[2] = lsDomain<NumericType, D>::BoundaryType::INFINITE_BOUNDARY;

  auto dom = lsSmartPointer<lsDomain<NumericType, D>>::New(bounds, boundaryCons,
                                                           gridDelta);

  {
    NumericType origin[3] = {0., 0., 0.};
    NumericType planeNormal[3] = {0., 0., 1.};
    auto plane =
        lsSmartPointer<lsPlane<NumericType, D>>::New(origin, planeNormal);
    lsMakeGeometry<NumericType, D>(dom, plane).apply();
  }

  {
    auto trench = lsSmartPointer<lsDomain<NumericType, D>>::New(
        bounds, boundaryCons, gridDelta);
    // make -x and +x greater than domain for numerical stability
    NumericType minCorner[D] = {-extent - 1, -extent / 4.f, -30.};
    NumericType maxCorner[D] = {extent + 1, extent / 4.f, 1.};
    auto box = lsSmartPointer<lsBox<NumericType, D>>::New(minCorner, maxCorner);
    lsMakeGeometry<NumericType, D>(trench, box).apply();
    // Create trench geometry
    lsBooleanOperation<NumericType, D>(
        dom, trench, lsBooleanOperationEnum::RELATIVE_COMPLEMENT)
        .apply();
  }

  rtTrace<particle<NumericType>, reflection, D> rayTracer;
  // rayTracer.setPowerCosineDirection(2.);
  auto newLayer = lsSmartPointer<lsDomain<NumericType, D>>::New(dom);

  std::cout << "Advecting" << std::endl;
  lsAdvect<NumericType, D> advectionKernel;

  advectionKernel.insertNextLevelSet(dom);
  advectionKernel.insertNextLevelSet(newLayer);

  // advectionKernel.setAdvectionTime(4.);
  // unsigned counter = 1;
  for (NumericType time = 0; time < 7.;
       time += advectionKernel.getAdvectedTime()) {
    rayTracer.setDomain(newLayer, gridDelta * 0.5 * std::sqrt(3) * (1 + eps));
    rayTracer.apply();
    auto mcestimates = rayTracer.getMcEstimates();
    auto velocities =
        lsSmartPointer<velocityField>::New(std::move(mcestimates));
    advectionKernel.setVelocityField(velocities);
    advectionKernel.apply();

    // auto meshAdvect = lsSmartPointer<lsMesh<NumericType>>::New();
    // lsToSurfaceMesh<NumericType, D>(newLayer, meshAdvect).apply();
    // lsVTKWriter<NumericType>(meshAdvect, "trench-" + std::to_string(counter)
    // + ".vtk")
    //     .apply();

    // ++counter;

    std::cout << "Time " << time << std::endl;
  }

  auto meshAdvect = lsSmartPointer<lsMesh<NumericType>>::New();
  lsToSurfaceMesh<NumericType, D>(newLayer, meshAdvect).apply();
  lsVTKWriter<NumericType>(meshAdvect, "trench-advected.vtk").apply();

  lsToSurfaceMesh<NumericType, D>(dom, meshAdvect).apply();
  lsVTKWriter<NumericType>(meshAdvect, "trench-initial.vtk").apply();

  return 0;
}
