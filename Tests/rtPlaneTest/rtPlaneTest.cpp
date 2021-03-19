#include <algorithm>
#include <iostream>
#include <lsDomain.hpp>
#include <lsMakeGeometry.hpp>
#include <lsToDiskMesh.hpp>
#include <lsToSurfaceMesh.hpp>
#include <lsVTKWriter.hpp>
#include <numeric>
#include <rtParticle.hpp>
#include <rtReflection.hpp>
#include <rtTrace.hpp>

using NumericType = float;

int main() {
  constexpr int D = 3;

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
    NumericType origin[D] = {0., 0., 0.};
    NumericType planeNormal[D] = {0., 0., 1.};
    auto plane =
        lsSmartPointer<lsPlane<NumericType, D>>::New(origin, planeNormal);
    lsMakeGeometry<NumericType, D>(dom, plane).apply();
  }

  size_t numRays;
  bool success = true;
  const NumericType eps = 1e-1;
  rtTrace<rtParticle1, rtDiffuseReflection, D> rayTracer(dom, gridDelta);
  rayTracer.setNumberOfRays(2000);
  rayTracer.setBoundaryY(
      rtTrace<rtParticle1, rtDiffuseReflection, D>::rtBoundary::REFLECTIVE);
  rayTracer.setBoundaryX(
      rtTrace<rtParticle1, rtDiffuseReflection, D>::rtBoundary::REFLECTIVE);
  rayTracer.apply();

  auto hitcounts = rayTracer.getHitCounts();
  auto mcestimates = rayTracer.getMcEstimates();
  {
    auto mesh = lsSmartPointer<lsMesh<NumericType>>::New();
    lsToDiskMesh<NumericType, D>(dom, mesh).apply();
    auto points = mesh.get()->getNodes();
    numRays = points.size() * 2000;
  }

  double sum = std::accumulate(hitcounts.begin(), hitcounts.end(), 0.0);
  double mean = sum / hitcounts.size();

  std::vector<double> diff(hitcounts.size());
  std::transform(hitcounts.begin(), hitcounts.end(), diff.begin(),
                 [mean](double x) { return x - mean; });
  double sq_sum =
      std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
  double stdev = std::sqrt(sq_sum / hitcounts.size());

  if (stdev > std::sqrt(numRays)) {
    std::cout << "Standard deviation too big!" << std::endl;
    success = false;
  }

  if (!std::all_of(mcestimates.begin(), mcestimates.end(),
                   [eps](NumericType i) { return (1 - i) < eps; })) {
    std::cout << "Mc estimates not equally distributed!" << std::endl;
    success = false;
  }

  if (success)
    std::cout << "Success" << std::endl;

  return 0;
}
