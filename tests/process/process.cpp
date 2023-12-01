#include <psProcess.hpp>

int main() {

  auto domain = psSmartPointer<psDomain<double, 2>>::New();
  auto model = psSmartPointer<psProcessModel<double, 2>>::New();

  // constructors
  { psProcess<double, 2> process; }
  { psProcess<double, 2> process(domain); }
  { psProcess<double, 2> process(domain, model, 0.); }
}