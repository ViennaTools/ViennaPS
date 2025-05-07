#include <geometries/psMakeTrench.hpp>
#include <models/psIsotropicProcess.hpp>

#include <psDomain.hpp>
#include <psProcess.hpp>

#include <lsTestAsserts.hpp>
#include <vcTestAsserts.hpp>

namespace viennacore {

using namespace viennaps;

template <class NumericType, int D> void RunTest() {
  Logger::setLogLevel(LogLevel::WARNING);

  {
    auto domain = SmartPointer<Domain<NumericType, D>>::New();
    MakeTrench<NumericType, D>(domain, 1., 10., 10., 2.5, 5., 10., 1., false,
                               true, Material::Si)
        .apply();
    auto model =
        SmartPointer<IsotropicProcess<NumericType, D>>::New(1., Material::Mask);

    VC_TEST_ASSERT(model->getSurfaceModel());
    VC_TEST_ASSERT(model->getVelocityField());
    VC_TEST_ASSERT(model->getVelocityField()->getTranslationFieldOptions() ==
                   0);

    Process<NumericType, D>(domain, model, 2.).apply();

    VC_TEST_ASSERT(domain->getLevelSets().size() == 2);
    VC_TEST_ASSERT(domain->getMaterialMap());
    VC_TEST_ASSERT(domain->getMaterialMap()->size() == 2);
    LSTEST_ASSERT_VALID_LS(domain->getLevelSets().back(), NumericType, D);
  }

  {
    auto domain = SmartPointer<Domain<NumericType, D>>::New();
    MakeTrench<NumericType, D>(domain, 1., 10., 10., 2.5, 5., 10., 1., false,
                               true, Material::Si)
        .apply();
    std::vector<Material> maskMaterials(1, Material::Mask);
    auto model =
        SmartPointer<IsotropicProcess<NumericType, D>>::New(1., maskMaterials);

    VC_TEST_ASSERT(model->getSurfaceModel());
    VC_TEST_ASSERT(model->getVelocityField());
    VC_TEST_ASSERT(model->getVelocityField()->getTranslationFieldOptions() ==
                   0);

    Process<NumericType, D>(domain, model, 2.).apply();

    VC_TEST_ASSERT(domain->getLevelSets().size() == 2);
    VC_TEST_ASSERT(domain->getMaterialMap());
    VC_TEST_ASSERT(domain->getMaterialMap()->size() == 2);
    LSTEST_ASSERT_VALID_LS(domain->getLevelSets().back(), NumericType, D);
  }

  {
    auto domain = SmartPointer<Domain<NumericType, D>>::New();
    MakeTrench<NumericType, D>(domain, 1., 10., 10., 2.5, 5., 10., 1., false,
                               true, Material::Si)
        .apply();

    std::unordered_map<Material, NumericType> materialRates(
        {{Material::Mask, -0.1}, {Material::Si, -1.}});
    auto model =
        SmartPointer<IsotropicProcess<NumericType, D>>::New(materialRates);

    VC_TEST_ASSERT(model->getSurfaceModel());
    VC_TEST_ASSERT(model->getVelocityField());
    VC_TEST_ASSERT(model->getVelocityField()->getTranslationFieldOptions() ==
                   0);

    Process<NumericType, D>(domain, model, 2.).apply();

    VC_TEST_ASSERT(domain->getLevelSets().size() == 2);
    VC_TEST_ASSERT(domain->getMaterialMap());
    VC_TEST_ASSERT(domain->getMaterialMap()->size() == 2);
    LSTEST_ASSERT_VALID_LS(domain->getLevelSets().back(), NumericType, D);
  }
}

} // namespace viennacore

int main() { VC_RUN_2D_TESTS }
