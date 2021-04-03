#include <iostream>
#include <lsAdvect.hpp>
#include <lsBooleanOperation.hpp>
#include <lsDomain.hpp>
#include <lsMakeGeometry.hpp>
#include <lsToSurfaceMesh.hpp>
#include <lsVTKWriter.hpp>
#include <rtParticle.hpp>
#include <rtReflection.hpp>
#include <rtTrace.hpp>

using NumericType = float;

int main() {
  constexpr int D = 3;

  omp_set_num_threads(4);

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

  // Create trench geometry
  {
    auto trench = lsSmartPointer<lsDomain<NumericType, D>>::New(
        bounds, boundaryCons, gridDelta);
    NumericType minCorner[D] = {-extent - 1, -extent / 4.f, -30.};
    NumericType maxCorner[D] = {extent + 1, extent / 4.f, 1.};
    auto box = lsSmartPointer<lsBox<NumericType, D>>::New(minCorner, maxCorner);
    lsMakeGeometry<NumericType, D>(trench, box).apply();
    lsBooleanOperation<NumericType, D>(
        dom, trench, lsBooleanOperationEnum::RELATIVE_COMPLEMENT)
        .apply();
  }

  auto mesh = lsSmartPointer<lsMesh<NumericType>>::New();
  lsToSurfaceMesh<NumericType, D>(dom, mesh).apply();
  lsVTKWriter<NumericType>(mesh, "trench-initial.vtk").apply();

  rtTrace<rtParticle1, rtDiffuseReflection, D> rayTracer(dom);
  rayTracer.setCosinePower(2.);
  rayTracer.setNumberOfRaysPerPoint(100);

  lsAdvect<NumericType, D> advectionKernel;
  advectionKernel.insertNextLevelSet(dom);
  advectionKernel.setVelocityField(rayTracer.getVelocityField());

  for (NumericType time = 0; time < 7.;
       time += advectionKernel.getAdvectedTime()) {
    std::cout << "Ray tracing ... " << std::endl;
    rayTracer.apply();

    std::cout << "Advecting ... " << std::endl;
    advectionKernel.apply();

    std::cout << "Time: " << time << std::endl;
  }

  lsToSurfaceMesh<NumericType, D>(dom, mesh).apply();
  lsVTKWriter<NumericType>(mesh, "trench-advected.vtk").apply();

  return 0;
}