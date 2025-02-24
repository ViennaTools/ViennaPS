#include <geometries/psMakePlane.hpp>
#include <lsTestAsserts.hpp>
#include <psDomain.hpp>
#include <vcTestAsserts.hpp>

namespace viennacore {

using namespace viennaps;

template <class NumericType, int D> void RunTest() {
  auto domain = SmartPointer<Domain<NumericType, D>>::New();

  MakePlane<NumericType, D>(domain, 1., 10., 10., 1., true, Material::Si)
      .apply();

  VC_TEST_ASSERT(domain->getLevelSets().size() == 1);
  VC_TEST_ASSERT(domain->getMaterialMap());
  VC_TEST_ASSERT(domain->getMaterialMap()->size() == 1);

  MakePlane<NumericType, D>(domain, 5., Material::Si, true).apply();

  VC_TEST_ASSERT(domain->getLevelSets().size() == 2);
  VC_TEST_ASSERT(domain->getMaterialMap()->size() == 2);

  LSTEST_ASSERT_VALID_LS(domain->getLevelSets().back(), NumericType, D);

  MakePlane<NumericType, D>(domain, 5., Material::Si, false).apply();

  VC_TEST_ASSERT(domain->getLevelSets().size() == 1);
  VC_TEST_ASSERT(domain->getMaterialMap()->size() == 1);
}

} // namespace viennacore

int main() { VC_RUN_ALL_TESTS }
