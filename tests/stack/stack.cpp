#include <geometries/psMakeStack.hpp>
#include <lsTestAsserts.hpp>
#include <psDomain.hpp>
#include <vcTestAsserts.hpp>

using namespace viennaps;

template <class NumericType, int D> void vtRunTest() {
  auto domain = SmartPointer<Domain<NumericType, D>>::New();

  MakeStack<NumericType, D>(domain, 1., 10., 10., 3 /*num layers*/, 3., 2., 0.,
                            0., 10, true)
      .apply();

  VC_TEST_ASSERT(domain->getLevelSets());
  VC_TEST_ASSERT(domain->getLevelSets()->size() == 5);
  VC_TEST_ASSERT(domain->getMaterialMap());

  LSTEST_ASSERT_VALID_LS(domain->getLevelSets()->back(), NumericType, D);
}

int main() { VC_RUN_ALL_TESTS }
