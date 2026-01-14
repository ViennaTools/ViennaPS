#include <geometries/psMakeHole.hpp>
#include <geometries/psMakeTrench.hpp>
#include <lsTestAsserts.hpp>
#include <models/psCSVFileProcess.hpp>
#include <process/psProcess.hpp>
#include <psDomain.hpp>
#include <vcTestAsserts.hpp>

#include <filesystem>

namespace viennacore {

using namespace viennaps;

template <typename NumericType, int D>
void writeCSV(const std::string &filename, bool etch = false) {
  std::ofstream out(filename);
  if (!out.is_open()) {
    std::cerr << "Error: Could not open file for writing: " << filename
              << std::endl;
    return;
  }

  std::mt19937 rng(42);
  std::uniform_real_distribution<NumericType> dist(1.0, 2.0); // rate range

  const int numPoints = 51;
  const NumericType minCoord = -50.0;
  const NumericType maxCoord = 50.0;
  const NumericType step = (maxCoord - minCoord) / (numPoints - 1);

  if constexpr (D == 2) {
    for (int i = 0; i < numPoints; ++i) {
      NumericType x = minCoord + i * step;
      NumericType rate = dist(rng);
      out << x << "," << (etch ? -rate : rate) << "\n";
    }
  } else if constexpr (D == 3) {
    for (int i = 0; i < numPoints; ++i) {
      NumericType x = minCoord + i * step;
      for (int j = 0; j < numPoints; ++j) {
        NumericType y = minCoord + j * step;
        NumericType rate = dist(rng);
        out << x << "," << y << "," << (etch ? -rate : rate) << "\n";
      }
    }
  }

  out.close();
}

template <typename NumericType, int D> void RunTest() {
  Logger::setLogLevel(LogLevel::WARNING);

  std::filesystem::create_directory("test_csv");
  const std::string csvPath = "test_csv/rates" + std::to_string(D) + "D.csv";

  for (bool etch : {false, true}) {
    writeCSV<NumericType, D>(csvPath, etch);

    for (const std::string &modeStr : {"linear", "idw", "custom"}) {
      std::cout << "[CSVFileProcess] Test: " << modeStr << " | Etch: " << etch
                << " | Dim: " << D << "\n";
      auto domain = Domain<NumericType, D>::New();

      if constexpr (D == 2)
        MakeTrench<NumericType, D>(domain, 1., 10., 10., 2.5, 5., 10., 0.0,
                                   false, etch, Material::Si)
            .apply();
      else
        MakeHole<NumericType, D>(domain, 1., 10., 10., 2.5, 5., 10., 0.0, false,
                                 etch, Material::Si, HoleShape::FULL)
            .apply();

      Vec2D<NumericType> offset{};
      auto direction = Vec3D<NumericType>{0., 0., 0.};
      direction[D - 1] = -1.;

      auto model = SmartPointer<CSVFileProcess<NumericType, D>>::New(
          csvPath, direction, offset);

      model->setInterpolationMode(modeStr);

      if (modeStr == "custom") {
        model->setCustomInterpolator([](const Vec3D<NumericType> &coord) {
          return 1.0 + 0.5 * std::sin(coord[0] + coord[1]);
        });
      } else if (modeStr == "idw") {
        model->setIDWNeighbors(4);
      }

      if (!etch)
        domain->duplicateTopLevelSet(Material::SiO2);

      VC_TEST_ASSERT(model->getSurfaceModel());
      VC_TEST_ASSERT(model->getVelocityField());

      Process<NumericType, D>(domain, model, 1.0).apply();

      VC_TEST_ASSERT(domain->getLevelSets().size() == 2);
      VC_TEST_ASSERT(domain->getMaterialMap());
      VC_TEST_ASSERT(domain->getMaterialMap()->size() == 2);
      LSTEST_ASSERT_VALID_LS(domain->getLevelSets().back(), NumericType, D);
    }
  }
  std::filesystem::remove_all("test_csv");
}

} // namespace viennacore

int main() { VC_RUN_ALL_TESTS }
