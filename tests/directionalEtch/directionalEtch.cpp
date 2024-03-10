#include <psDirectionalEtching.hpp>
#include <psDomain.hpp>
#include <psMakeTrench.hpp>
#include <psProcess.hpp>
#include <psTestAssert.hpp>

template <class NumericType, int D> void psRunTest() {
  psLogger::setLogLevel(psLogLevel::WARNING);

  {
    auto domain = psSmartPointer<psDomain<NumericType, D>>::New();
    psMakeTrench<NumericType, D>(domain, 1., 10., 10., 2.5, 5., 10., 1., false,
                                 true, psMaterial::Si)
        .apply();
    std::array<NumericType, 3> direction{0., 0., 0.};
    direction[D - 1] = -1.;
    auto model = psSmartPointer<psDirectionalEtching<NumericType, D>>::New(
        direction, 1., 0., psMaterial::Mask);

    PSTEST_ASSERT(model->getSurfaceModel());
    PSTEST_ASSERT(model->getVelocityField());
    PSTEST_ASSERT(model->getVelocityField()->getTranslationFieldOptions() == 0);

    psProcess<NumericType, D>(domain, model, 2.).apply();

    PSTEST_ASSERT(domain->getLevelSets());
    PSTEST_ASSERT(domain->getLevelSets()->size() == 2);
    PSTEST_ASSERT(domain->getMaterialMap());
    PSTEST_ASSERT(domain->getMaterialMap()->size() == 2);
    LSTEST_ASSERT_VALID_LS(domain->getLevelSets()->back(), NumericType, D);
  }

  {
    auto domain = psSmartPointer<psDomain<NumericType, D>>::New();
    psMakeTrench<NumericType, D>(domain, 1., 10., 10., 2.5, 5., 10., 1., false,
                                 true, psMaterial::Si)
        .apply();
    std::vector<psMaterial> maskMaterials(1, psMaterial::Mask);
    std::array<NumericType, 3> direction{0., 0., 0.};
    direction[D - 1] = -1.;
    auto model = psSmartPointer<psDirectionalEtching<NumericType, D>>::New(
        direction, 1., 0., maskMaterials);

    PSTEST_ASSERT(model->getSurfaceModel());
    PSTEST_ASSERT(model->getVelocityField());
    PSTEST_ASSERT(model->getVelocityField()->getTranslationFieldOptions() == 0);

    psProcess<NumericType, D>(domain, model, 2.).apply();

    PSTEST_ASSERT(domain->getLevelSets());
    PSTEST_ASSERT(domain->getLevelSets()->size() == 2);
    PSTEST_ASSERT(domain->getMaterialMap());
    PSTEST_ASSERT(domain->getMaterialMap()->size() == 2);
    LSTEST_ASSERT_VALID_LS(domain->getLevelSets()->back(), NumericType, D);
  }
}

int main() { PSRUN_ALL_TESTS }