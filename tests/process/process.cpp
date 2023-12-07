#include <psProcess.hpp>
#include <psTestAssert.hpp>

template <class NumericType, int D> void psRunTest() {

  auto domain = psSmartPointer<psDomain<NumericType, D>>::New();
  auto model = psSmartPointer<psProcessModel<NumericType, D>>::New();

  // constructors
  { psProcess<NumericType, D> process; }
  { psProcess<NumericType, D> process(domain); }
  { psProcess<NumericType, D> process(domain, model, 0.); }
}

int main() { PSRUN_ALL_TESTS }