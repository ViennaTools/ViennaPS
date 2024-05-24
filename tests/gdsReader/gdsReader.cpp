#include <psGDSReader.hpp>
#include <vcTestAsserts.hpp>

namespace viennacore {

using namespace viennaps;

template <class NumericType, int D> void RunTest() {
  const NumericType gridDelta = 0.01;
  lsBoundaryConditionEnum<D> boundaryConditions[D] = {
      lsBoundaryConditionEnum<D>::REFLECTIVE_BOUNDARY,
      lsBoundaryConditionEnum<D>::REFLECTIVE_BOUNDARY,
      lsBoundaryConditionEnum<D>::INFINITE_BOUNDARY};
  auto mask = SmartPointer<GDSGeometry<NumericType, D>>::New(gridDelta);
  mask->setBoundaryConditions(boundaryConditions);
  GDSReader<NumericType, D> reader(mask, "mask.gds");
}

} // namespace viennacore

int main() { VC_RUN_3D_TESTS }
