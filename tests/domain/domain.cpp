#include <psDomain.hpp>

int main() {
  constexpr int D = 2;

  {
    // default constructor
    auto domain = psSmartPointer<psDomain<double, D>>::New();
  }

  {
    // single LS constructor
    // auto ls = psSmartPointer<lsDomain<double, D>>::New();
    // auto domain = psSmartPointer<psDomain<double, D>>::New(ls);
  }
}