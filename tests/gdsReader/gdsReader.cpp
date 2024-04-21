#include <psGDSReader.hpp>
#include <psTestAssert.hpp>

template <class NumericType, int D> void psRunTest() {
  const NumericType gridDelta = 0.01;
  lsBoundaryConditionEnum<D> boundaryConditions[D] = {
      lsBoundaryConditionEnum<D>::REFLECTIVE_BOUNDARY,
      lsBoundaryConditionEnum<D>::REFLECTIVE_BOUNDARY,
      lsBoundaryConditionEnum<D>::INFINITE_BOUNDARY};
  auto mask = psSmartPointer<psGDSGeometry<NumericType, D>>::New(gridDelta);
  mask->setBoundaryConditions(boundaryConditions);
  psGDSReader<NumericType, D> reader(mask, "mask.gds");
}

int main() { PSRUN_3D_TESTS }
