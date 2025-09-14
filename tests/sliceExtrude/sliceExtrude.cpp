#include <geometries/psMakeTrench.hpp>
#include <psDomain.hpp>
#include <psExtrude.hpp>

using namespace viennaps;

int main() {
  constexpr double gridDelta = 0.013;
  constexpr double xExtent = 1.0;
  constexpr double yExtent = 1.0;

  auto domain2D = Domain<double, 2>::New(gridDelta, xExtent,
                                         BoundaryType::REFLECTIVE_BOUNDARY);

  MakeTrench<double, 2>(domain2D, 0.5, 0.5).apply();

  domain2D->saveSurfaceMesh("sliceExtrude2D.vtp");

  auto domain3D = Domain<double, 3>::New();
  Extrude<double> extruder;
  extruder.setInputDomain(domain2D);
  extruder.setOutputDomain(domain3D);
  extruder.setExtent({0.0, yExtent});
  extruder.setBoundaryConditions({BoundaryType::REFLECTIVE_BOUNDARY,
                                  BoundaryType::REFLECTIVE_BOUNDARY,
                                  BoundaryType::INFINITE_BOUNDARY});

  extruder.apply();

  domain3D->saveSurfaceMesh("sliceExtrude3D.vtp");
}