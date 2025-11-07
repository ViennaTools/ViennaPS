#include <geometries/psMakeTrench.hpp>
#include <lsTestAsserts.hpp>
#include <psDomain.hpp>
#include <vcTestAsserts.hpp>

namespace viennacore {

using namespace viennaps;

template <class NumericType, int D> void RunTest() {
  {
    auto domain = Domain<NumericType, D>::New();

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
    // Logger::setLogLevel(LogLevel::DEBUG);

    auto domain = Domain<NumericType, D>::New(.5, 10., 10.,
                                              BoundaryType::PERIODIC_BOUNDARY);
    MakeTrench<NumericType, D>(domain, 5., 15., 40., 5., -10., false,
                               Material::SiO2)
        .apply();
    // domain->saveSurfaceMesh("trench_2_" + std::to_string(D) + "D");

    VC_TEST_ASSERT(domain->getLevelSets().size() == 2);
    VC_TEST_ASSERT(domain->getMaterialMap());
    VC_TEST_ASSERT(domain->getMaterialMap()->size() == 2);

    LSTEST_ASSERT_VALID_LS(domain->getLevelSets().back(), NumericType, D);
  }

  {
    auto domain = Domain<NumericType, D>::New(
        1.0, 30., 30., BoundaryType::REFLECTIVE_BOUNDARY);

    std::vector<typename MakeTrench<NumericType, D>::MaterialLayer> layers = {
        {10., 3., 5., Material::Si, false},
        {5., 2., 10., Material::Mask, true},
        {15., 4., 0., Material::SiO2, false}};

    MakeTrench<NumericType, D>(domain, layers, false).apply();
    domain->saveSurfaceMesh("trench_3_" + std::to_string(D) + "D");

    VC_TEST_ASSERT(domain->getLevelSets().size() == 3);
    VC_TEST_ASSERT(domain->getMaterialMap());
    VC_TEST_ASSERT(domain->getMaterialMap()->size() == 3);

    LSTEST_ASSERT_VALID_LS(domain->getLevelSets().back(), NumericType, D);
  }
}

} // namespace viennacore

int main() { VC_RUN_ALL_TESTS }
