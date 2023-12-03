#include <psProcessModel.hpp>
#include <psTestAssert.hpp>

#include <psDirectionalEtching.hpp>
#include <psFluorocarbonEtching.hpp>
#include <psGeometricDistributionModels.hpp>
#include <psIsotropicProcess.hpp>
#include <psOxideRegrowth.hpp>
#include <psPlasmaDamage.hpp>
#include <psSF6O2Etching.hpp>
#include <psTEOSDeposition.hpp>
#include <psWetEtching.hpp>

template <class NumericType, int D> void runTest() {
  // default constructors
  { auto model = psSmartPointer<psProcessModel<NumericType, D>>::New(); }

  // directional etching
  {
    const std::array<NumericType, 3> dir = {0.};
    auto model = psSmartPointer<psDirectionalEtching<NumericType, D>>::New(dir);
    PSTEST_ASSERT(model->getSurfaceModel());
    PSTEST_ASSERT(model->getVelocityField());
    PSTEST_ASSERT(model->getVelocityField()->getTranslationFieldOptions() == 0);
  }

  // fluorocarbon etching
  {
    auto model = psSmartPointer<psFluorocarbonEtching<NumericType, D>>::New(
        0., 0., 0., 0., 0.);
    PSTEST_ASSERT(model->getSurfaceModel());
    PSTEST_ASSERT(model->getVelocityField());
    PSTEST_ASSERT(model->getParticleTypes());
    PSTEST_ASSERT(model->getParticleTypes()->size() == 3);
  }

  // geometric models
  {
    auto model =
        psSmartPointer<psSphereDistribution<NumericType, D>>::New(0., 0.);
    PSTEST_ASSERT(model->getGeometricModel());
  }

  {
    const std::array<double, 3> axes = {0.};
    auto model =
        psSmartPointer<psBoxDistribution<NumericType, D>>::New(axes, 0.);
    PSTEST_ASSERT(model->getGeometricModel());
  }

  // isotropic model
  {
    auto model = psSmartPointer<psIsotropicProcess<NumericType, D>>::New(0.);
    PSTEST_ASSERT(model->getSurfaceModel());
    PSTEST_ASSERT(model->getVelocityField());
    PSTEST_ASSERT(model->getVelocityField()->getTranslationFieldOptions() == 0);
  }

  // oxide regrowth
  {
    auto model = psSmartPointer<psOxideRegrowth<NumericType, D>>::New(
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.);
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
        psSmartPointer<psSF6O2Etching<NumericType, D>>::New(0., 0., 0., 0., 0.);
    PSTEST_ASSERT(model->getSurfaceModel());
    PSTEST_ASSERT(model->getVelocityField());
    PSTEST_ASSERT(model->getParticleTypes());
    PSTEST_ASSERT(model->getParticleTypes()->size() == 3);
  }
}

int main() {
  runTest<double, 2>();
  runTest<double, 3>();
  runTest<float, 2>();
  runTest<float, 3>();
}