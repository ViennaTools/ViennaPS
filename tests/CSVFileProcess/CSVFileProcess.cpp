#include <geometries/psMakeHole.hpp>
#include <geometries/psMakeTrench.hpp>
#include <lsTestAsserts.hpp>
#include <models/psCSVFileProcess.hpp>
#include <psDomain.hpp>
#include <psProcess.hpp>
#include <vcTestAsserts.hpp>

#include <filesystem>
#include <fstream>
#include <random>

#ifdef _WIN32
#include <chrono>
#include <thread>
#endif

namespace viennacore {

using namespace viennaps;

template <typename NumericType, int D>
void writeCSV(const std::string &filename, bool etch = false) {
  std::ofstream out(filename);
  std::mt19937 rng(42); // Fixed seed for reproducibility
  std::uniform_real_distribution<NumericType> dist(1.0, 2.0); // Rate range

  const int numPoints = 100; // Number of samples along each axis
  const NumericType minCoord = -50.0;
  const NumericType maxCoord = 50.0;
  const NumericType step = (maxCoord - minCoord) / (numPoints - 1);

  if constexpr (D == 2) {
    out << "x,rate\n";
    for (int i = 0; i < numPoints; ++i) {
      NumericType x = minCoord + i * step;
      NumericType rate = dist(rng);
      out << x << "," << (etch ? -rate : rate) << std::endl;
    }
  } else {
    out << "x,y,rate\n";
    for (int i = 0; i < numPoints; ++i) {
      NumericType x = minCoord + i * step;
      for (int j = 0; j < numPoints; ++j) {
        NumericType y = minCoord + j * step;
        NumericType rate = dist(rng);
        out << x << "," << y << "," << (etch ? -rate : rate) << std::endl;
      }
    }
  }
  out.close();
}

template <typename NumericType, int D> void RunTest() {
  Logger::setLogLevel(LogLevel::WARNING);
  std::filesystem::create_directory("test_csv");
  const std::string csvPath =
      D == 2 ? "test_csv/rates2D.csv" : "test_csv/rates3D.csv";
  // for (bool etch : {false, true}) {
  for (bool etch : {false}) {
    writeCSV<NumericType, D>(csvPath, etch);

    for (bool useCustomInterp : {false, true}) {
      auto domain = SmartPointer<Domain<NumericType, D>>::New();

      if constexpr (D == 2)
        MakeTrench<NumericType, D>(domain, 1., 10., 10., 2.5, 5., 10., 0.0,
                                   false, etch, Material::Si)
            .apply();
      else
        MakeHole<NumericType, D>(domain, 1., 10., 10., 2.5, 5., 10., 0.0, false,
                                 etch, Material::Si, HoleShape::Full)
            .apply();

      std::array<NumericType, D == 2 ? 1 : 2> offset{};
      auto direction = Vec3D<NumericType>{0., 0., 0.};
      direction[D - 1] = -1.;

#ifdef _WIN32
      // Wait up to 200ms and retry opening
      bool fileReady = false;
      for (int i = 0; i < 10; ++i) {
        std::ifstream test(csvPath);
        if (test.good()) {
          fileReady = true;
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
      }
      VC_TEST_ASSERT(fileReady);
#endif

      auto model = SmartPointer<CSVFileProcess<NumericType, D>>::New(
          csvPath, direction, offset);

      if (useCustomInterp) {
        model->setInterpolationMode(
            impl::velocityFieldFromFile<NumericType, D>::Interpolation::CUSTOM);
        model->setCustomInterpolator([](const Vec3D<NumericType> &coord) {
          return 1.0 + 0.5 * std::sin(coord[0] + (D > 1 ? coord[1] : 0.0));
        });
      } else {
        model->setInterpolationMode(
            impl::velocityFieldFromFile<NumericType, D>::Interpolation::LINEAR);
      }
      if (!etch)
        domain->duplicateTopLevelSet(Material::SiO2);

      VC_TEST_ASSERT(model->getSurfaceModel());
      VC_TEST_ASSERT(model->getVelocityField());
      VC_TEST_ASSERT(model->getVelocityField()->getTranslationFieldOptions() ==
                     0);

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
