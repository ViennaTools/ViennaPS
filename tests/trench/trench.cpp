#include <geometries/psMakeTrench.hpp>
#include <lsTestAsserts.hpp>
#include <psDomain.hpp>
#include <vcTestAsserts.hpp>

namespace viennacore {

using namespace viennaps;

template <class NumericType, int D> void RunTest() {
  {

    auto domain = SmartPointer<Domain<NumericType, D>>::New();

    MakeTrench<NumericType, D>(domain, 1., 10., 10., 5., 5., 10., 1., false,
                               true, Material::Si)
        .apply();

    VC_TEST_ASSERT(domain->getLevelSets().size() == 2);
    VC_TEST_ASSERT(domain->getMaterialMap());
    VC_TEST_ASSERT(domain->getMaterialMap()->size() == 2);

    LSTEST_ASSERT_VALID_LS(domain->getLevelSets().back(), NumericType, D);
  }

  {
    auto domain =
        SmartPointer<Domain<NumericType, D>>::New(.2, 10., 10., false);
    MakeTrench<NumericType, D>(domain, 3., 5., 10., 2., 0., Material::SiO2)
        .apply();

    VC_TEST_ASSERT(domain->getLevelSets().size() == 2);
    VC_TEST_ASSERT(domain->getMaterialMap());
    VC_TEST_ASSERT(domain->getMaterialMap()->size() == 2);

    LSTEST_ASSERT_VALID_LS(domain->getLevelSets().back(), NumericType, D);
  }
}

} // namespace viennacore

int main() { VC_RUN_ALL_TESTS }
