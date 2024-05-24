#include <psProcessModel.hpp>
#include <vcTestAsserts.hpp>

#include <models/psAnisotropicProcess.hpp>
#include <models/psDirectionalEtching.hpp>
#include <models/psFluorocarbonEtching.hpp>
#include <models/psGeometricDistributionModels.hpp>
#include <models/psIsotropicProcess.hpp>
#include <models/psOxideRegrowth.hpp>
#include <models/psPlasmaDamage.hpp>
#include <models/psSF6O2Etching.hpp>
#include <models/psSingleParticleProcess.hpp>
#include <models/psTEOSDeposition.hpp>

namespace viennacore {

using namespace viennaps;

template <class NumericType, int D> void RunTest() {
  // default constructors
  { auto model = SmartPointer<ProcessModel<NumericType, D>>::New(); }

  // fluorocarbon etching
  {
    auto model = SmartPointer<FluorocarbonEtching<NumericType, D>>::New(
        1., 1., 1., 1., 1.);
    VC_TEST_ASSERT(model->getSurfaceModel());
    VC_TEST_ASSERT(model->getVelocityField());
    VC_TEST_ASSERT(model->getParticleTypes().size() == 3);
  }

  // geometric models
  {
    auto model = SmartPointer<SphereDistribution<NumericType, D>>::New(1., 1.);
    VC_TEST_ASSERT(model->getGeometricModel());
  }

  {
    const std::array<double, 3> axes = {1.};
    auto model = SmartPointer<BoxDistribution<NumericType, D>>::New(axes, 0.);
    VC_TEST_ASSERT(model->getGeometricModel());
  }

  // // oxide regrowth
  // {
  //   auto model = SmartPointer<psOxideRegrowth<NumericType, D>>::New(
  //       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.);
  //   VC_TEST_ASSERT(model->getSurfaceModel());
  //   VC_TEST_ASSERT(model->getVelocityField());
  //   VC_TEST_ASSERT(model->getAdvectionCallback());
  // }

  // // plasma damage
  // {
  //   auto model = SmartPointer<psPlasmaDamage<NumericType, D>>::New();
  //   VC_TEST_ASSERT(model->getAdvectionCallback());
  // }

  // SF6O2 etching
  {
    auto model =
        SmartPointer<SF6O2Etching<NumericType, D>>::New(1., 1., 1., 1., 1.);
    VC_TEST_ASSERT(model->getSurfaceModel());
    VC_TEST_ASSERT(model->getVelocityField());
    VC_TEST_ASSERT(model->getParticleTypes().size() == 3);
  }

  // single particle TEOS deposition
  {
    auto model = SmartPointer<TEOSDeposition<NumericType, D>>::New(1., 1., 1.);
    VC_TEST_ASSERT(model->getSurfaceModel());
    VC_TEST_ASSERT(model->getVelocityField());
    VC_TEST_ASSERT(model->getParticleTypes().size() == 1);
  }

  // multi particle TEOS deposition
  {
    auto model = SmartPointer<TEOSDeposition<NumericType, D>>::New(1., 1., 1.,
                                                                   1., 1., 1.);
    VC_TEST_ASSERT(model->getSurfaceModel());
    VC_TEST_ASSERT(model->getVelocityField());
    VC_TEST_ASSERT(model->getParticleTypes().size() == 2);
  }

  // anisotropic model
  {
    auto model = SmartPointer<AnisotropicProcess<NumericType, D>>::New(
        std::vector<std::pair<Material, NumericType>>{});
    VC_TEST_ASSERT(model->getSurfaceModel());
    VC_TEST_ASSERT(model->getVelocityField());
    VC_TEST_ASSERT(model->getVelocityField()->getTranslationFieldOptions() ==
                   0);
  }
}

} // namespace viennacore

int main() { VC_RUN_ALL_TESTS }
