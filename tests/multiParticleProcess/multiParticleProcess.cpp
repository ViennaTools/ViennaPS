#include <geometries/psMakeTrench.hpp>
#include <models/psMultiParticleProcess.hpp>

#include <lsTestAsserts.hpp>
#include <psDomain.hpp>
#include <psProcess.hpp>
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
    auto model = SmartPointer<MultiParticleProcess<NumericType, D>>::New();

    VC_TEST_ASSERT(model->getSurfaceModel());
    VC_TEST_ASSERT(model->getVelocityField());
    VC_TEST_ASSERT(model->getVelocityField()->getTranslationFieldOptions() ==
                   2);

    model->addNeutralParticle(1.);
    VC_TEST_ASSERT(model->getParticleTypes().size() == 1);

    model->addIonParticle(1000.);
    VC_TEST_ASSERT(model->getParticleTypes().size() == 2);

    model->setRateFunction(
        [](const std::vector<NumericType> &fluxes, const Material &material) {
          VC_TEST_ASSERT(fluxes.size() == 2);
          return material == Material::Si ? -(fluxes[0] + fluxes[1]) : 0;
        });

    Process<NumericType, D>(domain, model, 1.).apply();

    LSTEST_ASSERT_VALID_LS(domain->getLevelSets().back(), NumericType, D);
  }
}

} // namespace viennacore

int main() { VC_RUN_ALL_TESTS }
