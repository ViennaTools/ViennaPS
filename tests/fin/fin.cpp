#include <geometries/psMakeFin.hpp>
#include <lsTestAsserts.hpp>
#include <psDomain.hpp>
#include <vcTestAsserts.hpp>

namespace viennacore {

using namespace viennaps;

template <class NumericType, int D> void RunTest() {
  auto domain = SmartPointer<Domain<NumericType, D>>::New();

  MakeFin<NumericType, D>(domain, .2, 10., 10., 5., 5., 10., 1., false, true,
                          Material::Si)
      .apply();

  VC_TEST_ASSERT(domain->getLevelSets().size() == 2);
  VC_TEST_ASSERT(domain->getMaterialMap());
  VC_TEST_ASSERT(domain->getMaterialMap()->size() == 2);

  LSTEST_ASSERT_VALID_LS(domain->getLevelSets().back(), NumericType, D);

  domain->setup(1.0, 10, 10, false);
  MakeFin<NumericType, D>(domain, 5., 5., 5., 2., 0., false, Material::Si)
      .apply();

  VC_TEST_ASSERT(domain->getLevelSets().size() == 2);
  VC_TEST_ASSERT(domain->getMaterialMap());
  VC_TEST_ASSERT(domain->getMaterialMap()->size() == 2);

  LSTEST_ASSERT_VALID_LS(domain->getLevelSets().back(), NumericType, D);

  MakeFin<NumericType, D>(domain, 5., 5., 55., 2., 0., false, Material::Si)
      .apply();

  VC_TEST_ASSERT(domain->getLevelSets().size() == 1);
  VC_TEST_ASSERT(domain->getMaterialMap());
  VC_TEST_ASSERT(domain->getMaterialMap()->size() == 1);

  LSTEST_ASSERT_VALID_LS(domain->getLevelSets().back(), NumericType, D);
}
} // namespace viennacore

int main() { VC_RUN_ALL_TESTS }
