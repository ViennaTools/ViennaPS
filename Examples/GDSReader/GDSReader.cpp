#include <psGDSReader.hpp>

int main(int argc, char **argv) {
  using NumericType = double;
  constexpr int D = 3;

  auto gds = psSmartPointer<psGDSGeometry<NumericType, D>>::New();
  std::string file = "box.gds";

  psGDSReader<NumericType, D>(gds, file).apply();

  gds->print();

  return 0;
}