#include <geometries/psMakeHole.hpp>
#include <lsTestAsserts.hpp>
#include <psDomain.hpp>
#include <vcTestAsserts.hpp>

namespace viennacore {

using namespace viennaps;

template <class NumericType, int D> void RunTest() {
  auto domain = SmartPointer<Domain<NumericType, D>>::New();

  // Test with HoleShape::Full
  MakeHole<NumericType, D>(domain, 1., 10., 10., 2.5, 5., 10., 1., false, true,
                           Material::Si, HoleShape::Full)
      .apply();

  VC_TEST_ASSERT(domain->getLevelSets().size() == 2);
  VC_TEST_ASSERT(domain->getMaterialMap());
  VC_TEST_ASSERT(domain->getMaterialMap()->size() == 2);

  LSTEST_ASSERT_VALID_LS(domain->getLevelSets().back(), NumericType, D);

  // Test with HoleShape::Quarter
  domain->clear(); // Reset the domain for a new test
  MakeHole<NumericType, D>(domain, 1., 10., 10., 2.5, 5., 10., 1., false, true,
                           Material::Si, HoleShape::Quarter)
      .apply();

  VC_TEST_ASSERT(domain->getLevelSets().size() == 2);
  VC_TEST_ASSERT(domain->getMaterialMap());
  VC_TEST_ASSERT(domain->getMaterialMap()->size() == 2);

  LSTEST_ASSERT_VALID_LS(domain->getLevelSets().back(), NumericType, D);

  // Test with HoleShape::Half
  domain->clear(); // Reset the domain for a new test
  MakeHole<NumericType, D>(domain, 1., 10., 10., 2.5, 5., 10., 1., false, true,
                           Material::Si, HoleShape::Half)
      .apply();

  VC_TEST_ASSERT(domain->getLevelSets().size() == 2);
  VC_TEST_ASSERT(domain->getMaterialMap());
  VC_TEST_ASSERT(domain->getMaterialMap()->size() == 2);

  LSTEST_ASSERT_VALID_LS(domain->getLevelSets().back(), NumericType, D);
}

} // namespace viennacore

int main() { VC_RUN_ALL_TESTS }
