#include <psProcess.hpp>
#include <vcTestAsserts.hpp>

namespace viennacore {

using namespace viennaps;

template <class NumericType, int D> void RunTest() {

  auto domain = SmartPointer<Domain<NumericType, D>>::New();
  auto model = SmartPointer<ProcessModel<NumericType, D>>::New();

  // constructors
  {
    Process<NumericType, D> process;
  }
  {
    Process<NumericType, D> process(domain);
  }
  {
    Process<NumericType, D> process(domain, model, 0.);
  }
}

} // namespace viennacore

int main() { VC_RUN_ALL_TESTS }