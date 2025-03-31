#include <geometries/psMakeStack.hpp>
#include <lsTestAsserts.hpp>
#include <psDomain.hpp>
#include <vcTestAsserts.hpp>

namespace viennacore {

using namespace viennaps;

template <class NumericType, int D> void RunTest() {
  Logger::setLogLevel(LogLevel::DEBUG);
#ifdef _OPENMP
  omp_set_num_threads(1);
#endif
  auto domain = SmartPointer<Domain<NumericType, D>>::New();
  MakeStack<NumericType, D>(domain, 1.0, 10., 10., 5 /*num layers*/, 3., 2., 2.,
                            2., 0., true)
      .apply();
  domain->saveLevelSetMesh("stack_1_" + std::to_string(D) + "D");

  VC_TEST_ASSERT(domain->getLevelSets().size() == 6);
  VC_TEST_ASSERT(domain->getMaterialMap());

  LSTEST_ASSERT_VALID_LS(domain->getLevelSets().back(), NumericType, D);

  domain->setup(1.0, 10., 10., BoundaryType::REFLECTIVE_BOUNDARY);
  MakeStack<NumericType, D>(domain, 5, 1., 2., 0., 5., 3., 10., true).apply();
  domain->saveLevelSetMesh("stack_2_" + std::to_string(D) + "D");

  VC_TEST_ASSERT(domain->getLevelSets().size() == 7);
  VC_TEST_ASSERT(domain->getMaterialMap());

  LSTEST_ASSERT_VALID_LS(domain->getLevelSets().back(), NumericType, D);
}

} // namespace viennacore

int main() { VC_RUN_ALL_TESTS }
