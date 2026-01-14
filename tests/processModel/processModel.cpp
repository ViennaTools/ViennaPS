#include <process/psProcessModel.hpp>
#include <vcTestAsserts.hpp>

#include <models/psCF4O2Etching.hpp>
#include <models/psDirectionalProcess.hpp>
#include <models/psFluorocarbonEtching.hpp>
#include <models/psGeometricDistributionModels.hpp>
#include <models/psIsotropicProcess.hpp>
#include <models/psOxideRegrowth.hpp>
#include <models/psSF6O2Etching.hpp>
#include <models/psSingleParticleProcess.hpp>
#include <models/psTEOSDeposition.hpp>
#include <models/psWetEtching.hpp>

namespace viennacore {

using namespace viennaps;

template <class NumericType, int D> void RunTest() {
  units::Time::getInstance().setUnit("s");
  units::Length::getInstance().setUnit("nm");

  // default constructors
  { auto model = SmartPointer<ProcessModelCPU<NumericType, D>>::New(); }

  // fluorocarbon etching
  {
    auto params = FluorocarbonParameters<NumericType>();
    params.addMaterial({Material::Polymer, 2.});
    auto model = SmartPointer<FluorocarbonEtching<NumericType, D>>::New(params);
    VC_TEST_ASSERT(model->getSurfaceModel());
    VC_TEST_ASSERT(model->getVelocityField());
    VC_TEST_ASSERT(model->getParticleTypes().size() == 3);
  }

  // geometric models
  {
    NumericType radius = 1;
    auto model = SmartPointer<SphereDistribution<NumericType, D>>::New(radius);
    VC_TEST_ASSERT(model->getGeometricModel());
  }

  {
    const std::array<NumericType, 3> axes = {1.};
    auto model = SmartPointer<BoxDistribution<NumericType, D>>::New(axes);
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

  // CF4O2 etching
  {
    units::Time::getInstance().setUnit("s");
    units::Length::getInstance().setUnit("nm");
    auto model =
        SmartPointer<CF4O2Etching<NumericType, D>>::New(1., 1., 1., 1., 1., 1.);
    VC_TEST_ASSERT(model->getSurfaceModel());
    VC_TEST_ASSERT(model->getVelocityField());
    VC_TEST_ASSERT(model->getParticleTypes().size() == 4);
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

  // wet etching model
  {
    auto model = SmartPointer<WetEtching<NumericType, D>>::New(
        std::vector<std::pair<Material, NumericType>>{});
    VC_TEST_ASSERT(model->getSurfaceModel());
    VC_TEST_ASSERT(model->getVelocityField());
  }
}

} // namespace viennacore

int main() { VC_RUN_ALL_TESTS }
