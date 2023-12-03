#include <psDomain.hpp>
#include <psMakePlane.hpp>
#include <psTestAssert.hpp>

template <class NumericType, int D> void psRunTest() {
  auto domain = psSmartPointer<psDomain<NumericType, D>>::New();

  psMakePlane<NumericType, D>(domain, 1., 10., 10., 1., true, psMaterial::Si)
      .apply();

  PSTEST_ASSERT(domain->getLevelSets());
  PSTEST_ASSERT(domain->getLevelSets()->size() == 1);
  PSTEST_ASSERT(domain->getMaterialMap());
  PSTEST_ASSERT(domain->getMaterialMap()->size() == 1);

  psMakePlane<NumericType, D>(domain, 5., psMaterial::Si).apply();

  PSTEST_ASSERT(domain->getLevelSets()->size() == 2);
  PSTEST_ASSERT(domain->getMaterialMap()->size() == 2);
}

int main() { PSRUN_ALL_TESTS }