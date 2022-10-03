#include <psGDSReader.hpp>

int main(int argc, char **argv) {
  using NumericType = double;
  constexpr int D = 3;

  auto domain = psSmartPointer<psDomain<NumericType, D>>::New();
  std::string file = "box.gds";

  psGDSReader<NumericType, D>(domain, file).apply();

  return 0;
}