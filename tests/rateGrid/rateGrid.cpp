#include <lsTestAsserts.hpp>
#include <vcTestAsserts.hpp>

#include <psRateGrid.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>

namespace viennacore {

using namespace viennaps;

template <typename NumericType, int D>
void writeCSV(const std::string &filename) {
  std::ofstream out(filename);
  if (!out.is_open()) {
    std::cerr << "Error: Could not open file for writing: " << filename
              << std::endl;
    return;
  }

  std::mt19937 rng(42);
  std::uniform_real_distribution<NumericType> dist(0.5, 1.5); // rate range

  const int numPoints = 51;
  const NumericType minCoord = -5.0;
  const NumericType maxCoord = 5.0;
  const NumericType step = (maxCoord - minCoord) / (numPoints - 1);

  if constexpr (D == 2) {
    for (int i = 0; i < numPoints; ++i) {
      NumericType x = minCoord + i * step;
      NumericType rate = dist(rng);
      out << x << "," << rate << "\n";
    }
  } else if constexpr (D == 3) {
    for (int i = 0; i < numPoints; ++i) {
      NumericType x = minCoord + i * step;
      for (int j = 0; j < numPoints; ++j) {
        NumericType y = minCoord + j * step;
        NumericType rate = dist(rng);
        out << x << "," << y << "," << rate << "\n";
      }
    }
  }

  out.close();
}

template <typename NumericType, int D> void RunTest() {
  Logger::setLogLevel(LogLevel::WARNING);

  std::filesystem::create_directory("test_rategrid");
  const std::string csvPath =
      "test_rategrid/rates" + std::to_string(D) + "D.csv";

  writeCSV<NumericType, D>(csvPath);

  for (auto mode : {RateGrid<NumericType, D>::Interpolation::LINEAR,
                    RateGrid<NumericType, D>::Interpolation::IDW,
                    RateGrid<NumericType, D>::Interpolation::CUSTOM}) {

    RateGrid<NumericType, D> grid;
    VC_TEST_ASSERT(grid.loadFromCSV(csvPath));
    grid.setOffset({0., 0.});
    grid.setInterpolationMode(mode);

    if (mode == RateGrid<NumericType, D>::Interpolation::CUSTOM) {
      grid.setCustomInterpolator([](const Vec3D<NumericType> &coord) {
        return static_cast<NumericType>(1.0 + 0.1 * std::cos(coord[0] + coord[1]));
      });
    } else if (mode == RateGrid<NumericType, D>::Interpolation::IDW) {
      grid.setIDWNeighbors(4);
    }

    // Choose test coordinates within expected range
    std::mt19937 rng(42);
    std::uniform_real_distribution<NumericType> coordDom(-5.0, 5.0);
    for (int i = 0; i < 10; ++i) {
      Vec3D<NumericType> coord = {coordDom(rng), 0.0, 0.0};
      if constexpr (D == 3)
        coord[1] = coordDom(rng);

      NumericType result = grid.interpolate(coord);
      if (mode != RateGrid<NumericType, D>::Interpolation::CUSTOM)
        VC_TEST_ASSERT(result >= 0.5 && result <= 1.5);
      std::cout << "[RateGrid] Interpolation at (";
      for (int j = 0; j < D - 1; ++j) {
        std::cout << coord[j];
        if (j < D - 2)
          std::cout << ", ";
      }
      std::cout << ") = " << result << std::endl;
    }
  }
  std::filesystem::remove_all("test_rategrid");
}

} // namespace viennacore

int main() { VC_RUN_ALL_TESTS }
