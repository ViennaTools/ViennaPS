#include <geometries/psGeometryFactory.hpp>
#include <psDomain.hpp>

#include <vcTestAsserts.hpp>

namespace viennacore {
template <class NumericType, int D> void RunTest() {
  auto domain = viennaps::Domain<NumericType, D>::New(
      1.0, 10., 10.0, viennaps::BoundaryType::REFLECTIVE_BOUNDARY);

  viennaps::GeometryFactory<NumericType, D> geo(domain->getSetup());

  auto box = geo.makeBoxStencil({0.5, 2.0, 0.0}, 1.0, 2.5, 0.0);

  auto cylinder = geo.makeCylinderStencil({-1.0, 0.0, 0.0}, 0.5, 3.0, 10.0);

  auto substrate = geo.makeSubstrate(-1.0);

  auto mask = geo.makeMask(0.0, 2.0);
}
} // namespace viennacore

int main() { VC_RUN_3D_TESTS }