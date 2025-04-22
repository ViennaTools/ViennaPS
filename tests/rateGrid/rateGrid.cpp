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

  for (bool useCustomInterp : {false, true}) {
    RateGrid<NumericType, D> grid;
    VC_TEST_ASSERT(grid.loadFromCSV(csvPath));

    Vec2D<NumericType> offset = {0., 0.};
    grid.setOffset(offset);

    if (useCustomInterp) {
      grid.setInterpolationMode(
          RateGrid<NumericType, D>::Interpolation::CUSTOM);
      grid.setCustomInterpolator([](const Vec3D<NumericType> &coord) {
        return 1.0 + 0.1 * std::cos(coord[0] + coord[1]);
      });
    } else {
      grid.setInterpolationMode(
          RateGrid<NumericType, D>::Interpolation::LINEAR);
    }

    // Choose test coordinates within expected range
    Vec3D<NumericType> coord = {0.0, 0.0, 0.0};
    coord[0] = 0.2;
    if constexpr (D == 3)
      coord[1] = -0.3;

    NumericType result = grid.interpolate(coord);

    VC_TEST_ASSERT(result > 0.0); // Basic sanity check
    std::cout << "[RateGrid] Interpolation result: " << result << std::endl;
  }

  std::filesystem::remove_all("test_rategrid");
}

} // namespace viennacore

int main() { VC_RUN_ALL_TESTS }
