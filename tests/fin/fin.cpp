#include <geometries/psMakeFin.hpp>
#include <lsTestAsserts.hpp>
#include <psDomain.hpp>
#include <vcTestAsserts.hpp>

using namespace viennaps;

template <class NumericType, int D> void vtRunTest() {
  auto domain = SmartPointer<Domain<NumericType, D>>::New();

  MakeFin<NumericType, D>(domain, 1., 10., 10., 5., 5., 10., 1., false, true,
                          Material::Si)
      .apply();

  VC_TEST_ASSERT(domain->getLevelSets());
  VC_TEST_ASSERT(domain->getLevelSets()->size() == 2);
  VC_TEST_ASSERT(domain->getMaterialMap());
  VC_TEST_ASSERT(domain->getMaterialMap()->size() == 2);

  LSTEST_ASSERT_VALID_LS(domain->getLevelSets()->back(), NumericType, D);
}

int main() { VC_RUN_ALL_TESTS }
