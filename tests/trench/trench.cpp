#include <psDomain.hpp>
#include <psMakeTrench.hpp>
#include <psTestAssert.hpp>

template <class NumericType, int D> void psRunTest() {
  auto domain = psSmartPointer<psDomain<NumericType, D>>::New();

  psMakeTrench<NumericType, D>(domain, 1., 10., 10., 5., 5., 10., 1., false,
                               true, psMaterial::Si)
      .apply();

  PSTEST_ASSERT(domain->getLevelSets());
  PSTEST_ASSERT(domain->getLevelSets()->size() == 2);
  PSTEST_ASSERT(domain->getMaterialMap());
  PSTEST_ASSERT(domain->getMaterialMap()->size() == 2);

  LSTEST_ASSERT_VALID_LS(domain->getLevelSets()->back(), NumericType, D);
}

int main() { PSRUN_ALL_TESTS }