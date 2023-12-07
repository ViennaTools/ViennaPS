#include <psDomain.hpp>
#include <psMakeStack.hpp>
#include <psTestAssert.hpp>

template <class NumericType, int D> void psRunTest() {
  auto domain = psSmartPointer<psDomain<NumericType, D>>::New();

  psMakeStack<NumericType, D>(domain, 1., 10., 10., 3 /*num layers*/, 3., 2.,
                              0., 0., 10, true)
      .apply();

  PSTEST_ASSERT(domain->getLevelSets());
  PSTEST_ASSERT(domain->getLevelSets()->size() == 5);
  PSTEST_ASSERT(domain->getMaterialMap());

  LSTEST_ASSERT_VALID_LS(domain->getLevelSets()->back(), NumericType, D);
}

int main() { PSRUN_ALL_TESTS }