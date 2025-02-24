#include <geometries/psMakeTrench.hpp>
#include <lsTestAsserts.hpp>
#include <psDomain.hpp>
#include <vcTestAsserts.hpp>

namespace viennacore {

using namespace viennaps;

template <class NumericType, int D> void RunTest() {
  {
    auto domain = SmartPointer<Domain<NumericType, D>>::New();

    MakeTrench<NumericType, D>(domain, .5, 10., 10., 5., 5., 10., 1., false,
                               true, Material::Si)
        .apply();
    domain->saveSurfaceMesh("trench_1_" + std::to_string(D) + "D");

    VC_TEST_ASSERT(domain->getLevelSets().size() == 2);
    VC_TEST_ASSERT(domain->getMaterialMap());
    VC_TEST_ASSERT(domain->getMaterialMap()->size() == 2);

    LSTEST_ASSERT_VALID_LS(domain->getLevelSets().back(), NumericType, D);
  }

  {
    // Logger::getInstance().setLogLevel(LogLevel::DEBUG);

    auto domain = SmartPointer<Domain<NumericType, D>>::New(
        .5, 10., 10., BoundaryType::PERIODIC_BOUNDARY);
    MakeTrench<NumericType, D>(domain, 5., 15., 40., 5., -10., false,
                               Material::SiO2)
        .apply();
    // domain->saveSurfaceMesh("trench_2_" + std::to_string(D) + "D");

    VC_TEST_ASSERT(domain->getLevelSets().size() == 2);
    VC_TEST_ASSERT(domain->getMaterialMap());
    VC_TEST_ASSERT(domain->getMaterialMap()->size() == 2);

    LSTEST_ASSERT_VALID_LS(domain->getLevelSets().back(), NumericType, D);
  }
}

} // namespace viennacore

int main() { VC_RUN_ALL_TESTS }
