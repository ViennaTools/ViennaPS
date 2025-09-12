#include <geometries/psMakeTrench.hpp>
#include <models/psDirectionalProcess.hpp>

#include <lsTestAsserts.hpp>
#include <process/psProcess.hpp>
#include <psDomain.hpp>
#include <vcTestAsserts.hpp>

namespace viennacore {

using namespace viennaps;

template <class NumericType, int D> void RunTest() {
  Logger::setLogLevel(LogLevel::WARNING);

  {
    auto domain = Domain<NumericType, D>::New();
    MakeTrench<NumericType, D>(domain, 1., 10., 10., 2.5, 5., 10., 1., false,
                               true, Material::Si)
        .apply();
    Vec3D<NumericType> direction{0., 0., 0.};
    direction[D - 1] = -1.;
    auto model = SmartPointer<DirectionalProcess<NumericType, D>>::New(
        direction, 1., 0., Material::Mask);

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
    auto domain = Domain<NumericType, D>::New();
    MakeTrench<NumericType, D>(domain, 1., 10., 10., 2.5, 5., 10., 1., false,
                               true, Material::Si)
        .apply();

    typename DirectionalProcess<NumericType, D>::RateSet rateSet;
    rateSet.direction = Vec3D<NumericType>{0., 0., 0.};
    rateSet.direction[D - 1] = -1.;
    rateSet.directionalVelocity = -1.;
    rateSet.isotropicVelocity = 0.;
    rateSet.maskMaterials = std::vector<Material>{Material::Mask};
    rateSet.calculateVisibility = false;

    auto model = SmartPointer<DirectionalProcess<NumericType, D>>::New(rateSet);

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

int main() { VC_RUN_ALL_TESTS }
