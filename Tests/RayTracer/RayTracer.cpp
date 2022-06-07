#include <lsDomain.hpp>
#include <lsMakeGeometry.hpp>
#include <lsToDiskMesh.hpp>
#include <rayTrace.hpp>

using NumericType = float;

int main() {
  constexpr int D = 3;

  omp_set_num_threads(12);

  NumericType extent = 10;
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

  auto mesh = lsSmartPointer<lsMesh<NumericType>>::New();
  lsToDiskMesh<NumericType, D>(dom, mesh).apply();

  auto points = mesh->getNodes();
  auto normals = *mesh->getCellData().getVectorData("Normals");

  auto particle = std::make_unique<rayTestParticle<NumericType>>();
  rayTrace<NumericType, D> rayTracer;
  rayTracer.setGeometry(points, normals, gridDelta);
  rayTracer.setParticleType(particle);
  rayTracer.setNumberOfRaysPerPoint(10);
  rayTracer.apply();

  return 0;
}