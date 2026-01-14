#include <geometries/psMakeHole.hpp>
#include <lsTestAsserts.hpp>
#include <psDomain.hpp>
#include <vcTestAsserts.hpp>

namespace viennacore {

using namespace viennaps;

template <class NumericType, int D> void RunTest() {
#ifdef _OPENMP
  omp_set_num_threads(1);
#endif
  auto domain = Domain<NumericType, D>::New();

  // Test with HoleShape::Full
  MakeHole<NumericType, D>(domain, 1.0, 10., 10., 2.5, 5., 10., 1., false, true,
                           Material::Si, HoleShape::FULL)
      .apply();

  VC_TEST_ASSERT(domain->getLevelSets().size() == 2);
  VC_TEST_ASSERT(domain->getMaterialMap());
  VC_TEST_ASSERT(domain->getMaterialMap()->size() == 2);

  LSTEST_ASSERT_VALID_LS(domain->getLevelSets().back(), NumericType, D);

  //   // Test with HoleShape::Quarter
  domain->clear(); // Reset the domain for a new test
  MakeHole<NumericType, D>(domain, .5, 10., 10., 2.5, 5., 10., 1., false, true,
                           Material::Si, HoleShape::QUARTER)
      .apply();

  VC_TEST_ASSERT(domain->getLevelSets().size() == 2);
  VC_TEST_ASSERT(domain->getMaterialMap());
  VC_TEST_ASSERT(domain->getMaterialMap()->size() == 2);

  LSTEST_ASSERT_VALID_LS(domain->getLevelSets().back(), NumericType, D);

  // Test with HoleShape::Half
  domain->clear(); // Reset the domain for a new test
  MakeHole<NumericType, D>(domain, 1., 10., 10., 2.5, 5., 10., 1., false, true,
                           Material::Si, HoleShape::HALF)
      .apply();

  VC_TEST_ASSERT(domain->getLevelSets().size() == 2);
  VC_TEST_ASSERT(domain->getMaterialMap());
  VC_TEST_ASSERT(domain->getMaterialMap()->size() == 2);

  LSTEST_ASSERT_VALID_LS(domain->getLevelSets().back(), NumericType, D);
  // Logger::setLogLevel(LogLevel::DEBUG);

  domain->setup(0.5, 10., 10., BoundaryType::REFLECTIVE_BOUNDARY);
  MakeHole<NumericType, D>(domain, 2.5, 3., 10, 3., 10., HoleShape::QUARTER,
                           Material::Si)
      .apply();
  // domain->saveSurfaceMesh("hole" + std::to_string(D) + "D");

  VC_TEST_ASSERT(domain->getLevelSets().size() == 2);
  VC_TEST_ASSERT(domain->getMaterialMap());
  VC_TEST_ASSERT(domain->getMaterialMap()->size() == 2);

  LSTEST_ASSERT_VALID_LS(domain->getLevelSets().back(), NumericType, D);
}

} // namespace viennacore

int main() { VC_RUN_ALL_TESTS }
