#include <psProcessModel.hpp>
#include <psTestAssert.hpp>

#include <directionalEtching.hpp>
#include <fluorocarbonEtching.hpp>

int main() {

  // default constructors
  { auto model = psSmartPointer<psProcessModel<double, 2>>::New(); }
  { auto model = psSmartPointer<psProcessModel<double, 3>>::New(); }

  // directional etching
  {
    const std::array<double, 3> dir = {0.};
    auto model = psSmartPointer<DirectionalEtching<double, 2>>::New(dir);
    PSTEST_ASSERT(model->getSurfaceModel());
    PSTEST_ASSERT(model->getVelocityField());
  }

  // fluorocarbon etching
  {
    auto model =
        psSmartPointer<FluorocarbonEtching<double, 2>>::New(0., 0., 0., 0., 0.);
    PSTEST_ASSERT(model->getSurfaceModel());
    PSTEST_ASSERT(model->getVelocityField());
    PSTEST_ASSERT(model->getParticleTypes());
    PSTEST_ASSERT(model->getParticleTypes()->size() == 3);
  }
}