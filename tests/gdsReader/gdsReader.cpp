#include <psGDSReader.hpp>
#include <vcTestAsserts.hpp>

namespace viennacore {

using namespace viennaps;

template <class NumericType, int D> void RunTest() {
  const NumericType gridDelta = 0.01;
  viennals::BoundaryConditionEnum<D> boundaryConditions[D] = {
      viennals::BoundaryConditionEnum<D>::REFLECTIVE_BOUNDARY,
      viennals::BoundaryConditionEnum<D>::REFLECTIVE_BOUNDARY,
      viennals::BoundaryConditionEnum<D>::INFINITE_BOUNDARY};
  auto mask = SmartPointer<GDSGeometry<NumericType, D>>::New(gridDelta);
  mask->setBoundaryConditions(boundaryConditions);
  GDSReader<NumericType, D> reader(mask, "mask.gds");
}

} // namespace viennacore

int main() { VC_RUN_3D_TESTS }
