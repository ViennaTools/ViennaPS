#include <geometries/psMakeHole.hpp>
#include <geometries/psMakeTrench.hpp>
#include <lsTestAsserts.hpp>
#include <models/psCSVFileProcess.hpp>
#include <psDomain.hpp>
#include <psProcess.hpp>
#include <vcTestAsserts.hpp>

namespace viennacore {

using namespace viennaps;

template <typename NumericType, int D>
void RunTest() {
  Logger::setLogLevel(LogLevel::WARNING);

  for (bool etch : {false, true}) {
    // Select CSV file based on dimension and etch/deposit mode
    std::string csvPath = "rates" + std::to_string(D) + "D_" + (etch ? "etch" : "deposit") + ".csv";

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
}

} // namespace viennacore

int main() { VC_RUN_ALL_TESTS }
