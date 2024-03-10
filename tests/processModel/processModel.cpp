#include <psProcessModel.hpp>
#include <psTestAssert.hpp>

#include <psAnisotropicProcess.hpp>
#include <psDirectionalEtching.hpp>
#include <psFluorocarbonEtching.hpp>
#include <psGeometricDistributionModels.hpp>
#include <psIsotropicProcess.hpp>
#include <psOxideRegrowth.hpp>
#include <psPlasmaDamage.hpp>
#include <psSF6O2Etching.hpp>
#include <psSingleParticleProcess.hpp>
#include <psTEOSDeposition.hpp>

template <class NumericType, int D> void psRunTest() {
  // default constructors
  { auto model = psSmartPointer<psProcessModel<NumericType, D>>::New(); }

  // fluorocarbon etching
  {
    auto model = psSmartPointer<psFluorocarbonEtching<NumericType, D>>::New(
        1., 1., 1., 1., 1.);
    PSTEST_ASSERT(model->getSurfaceModel());
    PSTEST_ASSERT(model->getVelocityField());
    PSTEST_ASSERT(model->getParticleTypes());
    PSTEST_ASSERT(model->getParticleTypes()->size() == 3);
  }

  // geometric models
  {
    auto model =
        psSmartPointer<psSphereDistribution<NumericType, D>>::New(1., 1.);
    PSTEST_ASSERT(model->getGeometricModel());
  }

  {
    const std::array<double, 3> axes = {1.};
    auto model =
        psSmartPointer<psBoxDistribution<NumericType, D>>::New(axes, 0.);
    PSTEST_ASSERT(model->getGeometricModel());
  }

  // oxide regrowth
  {
    auto model = psSmartPointer<psOxideRegrowth<NumericType, D>>::New(
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.);
    PSTEST_ASSERT(model->getSurfaceModel());
    PSTEST_ASSERT(model->getVelocityField());
    PSTEST_ASSERT(model->getAdvectionCallback());
  }

  // plasma damage
  {
    auto model = psSmartPointer<psPlasmaDamage<NumericType, D>>::New();
    PSTEST_ASSERT(model->getAdvectionCallback());
  }

  // SF6O2 etching
  {
    auto model =
        psSmartPointer<psSF6O2Etching<NumericType, D>>::New(1., 1., 1., 1., 1.);
    PSTEST_ASSERT(model->getSurfaceModel());
    PSTEST_ASSERT(model->getVelocityField());
    PSTEST_ASSERT(model->getParticleTypes());
    PSTEST_ASSERT(model->getParticleTypes()->size() == 3);
  }

  // single particle TEOS deposition
  {
    auto model =
        psSmartPointer<psTEOSDeposition<NumericType, D>>::New(1., 1., 1.);
    PSTEST_ASSERT(model->getSurfaceModel());
    PSTEST_ASSERT(model->getVelocityField());
    PSTEST_ASSERT(model->getParticleTypes());
    PSTEST_ASSERT(model->getParticleTypes()->size() == 1);
  }

  // multi particle TEOS deposition
  {
    auto model = psSmartPointer<psTEOSDeposition<NumericType, D>>::New(
        1., 1., 1., 1., 1., 1.);
    PSTEST_ASSERT(model->getSurfaceModel());
    PSTEST_ASSERT(model->getVelocityField());
    PSTEST_ASSERT(model->getParticleTypes());
    PSTEST_ASSERT(model->getParticleTypes()->size() == 2);
  }

  // anisotropic model
  {
    auto model = psSmartPointer<psAnisotropicProcess<NumericType, D>>::New(
        std::vector<std::pair<psMaterial, NumericType>>{});
    PSTEST_ASSERT(model->getSurfaceModel());
    PSTEST_ASSERT(model->getVelocityField());
    PSTEST_ASSERT(model->getVelocityField()->getTranslationFieldOptions() == 0);
  }
}

int main() { PSRUN_ALL_TESTS }