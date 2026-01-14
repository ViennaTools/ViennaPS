#include <geometries/psMakeStack.hpp>
#include <psDomain.hpp>
#include <psExtrude.hpp>
#include <psSlice.hpp>

using namespace viennaps;

int main() {
  constexpr double gridDelta = 0.023;
  constexpr double xExtent = 1.0;
  constexpr double yExtent = 1.0;

  auto domain2D = Domain<double, 2>::New(gridDelta, xExtent,
                                         BoundaryType::REFLECTIVE_BOUNDARY);

  MakeStack<double, 2>(domain2D, 5, 0.1, 0.1, 0.2, 0.0, 0.0, 10).apply();

  domain2D->saveSurfaceMesh("sliceExtrude2D.vtp");

  auto domain3D = Domain<double, 3>::New();
  Extrude<double> extruder;
  extruder.setInputDomain(domain2D);
  extruder.setOutputDomain(domain3D);
  extruder.setExtent({-yExtent / 2., yExtent / 2.});
  extruder.setExtrusionAxis(1); // Extrude along y-axis
  extruder.setBoundaryConditions({BoundaryType::REFLECTIVE_BOUNDARY,
                                  BoundaryType::REFLECTIVE_BOUNDARY,
                                  BoundaryType::INFINITE_BOUNDARY});

  extruder.apply();

  domain3D->saveSurfaceMesh("sliceExtrude3D.vtp");

  auto slicedDomain2D = Domain<double, 2>::New();
  Slice<double> slicer;
  slicer.setInputDomain(domain3D);
  slicer.setOutputDomain(slicedDomain2D);
  slicer.setSliceDimension(1);  // Slice along y-axis
  slicer.setSlicePosition(0.0); // Slice at y = 0.0
  slicer.apply();

  slicedDomain2D->saveSurfaceMesh("sliceExtrude_sliced2D.vtp");

  return 0;
}