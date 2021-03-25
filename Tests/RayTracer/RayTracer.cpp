#include <iostream>
#include <lsAdvect.hpp>
#include <lsBooleanOperation.hpp>
#include <lsDomain.hpp>
#include <lsMakeGeometry.hpp>
#include <lsToMesh.hpp>
#include <lsToSurfaceMesh.hpp>
#include <lsVTKWriter.hpp>
#include <rtTrace.hpp>
#include <rtParticle.hpp>
#include <rtReflection.hpp>

using NumericType = float;

int main() {
  constexpr int D = 3;

  omp_set_num_threads(4);

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

  // create trench geometry
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
  auto newLayer = lsSmartPointer<lsDomain<NumericType, D>>::New(dom);

  rtTrace<rtParticle1, rtDiffuseReflection, D> rayTracer;
  rayTracer.setPowerCosineDirection(2.);
  rayTracer.setDomain(newLayer, gridDelta * 0.5 * std::sqrt(3) * (1 + eps));
  rayTracer.setNumberOfRays(100);

  lsAdvect<NumericType, D> advectionKernel;
  advectionKernel.insertNextLevelSet(dom);
  advectionKernel.insertNextLevelSet(newLayer);
  advectionKernel.setVelocityField(rayTracer.getVelocityField());

  std::cout << "Advecting" << std::endl;
  for (NumericType time = 0; time < 7.;
       time += advectionKernel.getAdvectedTime()) {
    rayTracer.apply();
    advectionKernel.apply();

    std::cout << "Time " << time << std::endl;
  }

  auto mesh = lsSmartPointer<lsMesh<NumericType>>::New();
  lsToSurfaceMesh<NumericType, D>(newLayer, mesh).apply();
  lsVTKWriter<NumericType>(mesh, "trench-advected.vtk").apply();

  lsToSurfaceMesh<NumericType, D>(dom, mesh).apply();
  lsVTKWriter<NumericType>(mesh, "trench-initial.vtk").apply();

  return 0;
}
