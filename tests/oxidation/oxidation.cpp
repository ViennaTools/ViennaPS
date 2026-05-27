#include <geometries/psMakePlane.hpp>
#include <models/psOxidation.hpp>
#include <process/psProcess.hpp>
#include <psDomain.hpp>

#include <vcTestAsserts.hpp>

#include <cmath>

namespace ps = viennaps;

using T = double;
constexpr int D = 2;

void testDealGroveEstimateWet1000C() {
  auto model = ps::SmartPointer<ps::Oxidation<T, D>>::New();
  model->setTemperature(1000.);
  model->setTime(0.1);
  model->setOxidant(ps::OxidantType::Wet);
  model->setPressure(1.);
  model->setOrientation(ps::SiliconOrientation::Si100);

  const T thickness = model->estimatePlanarOxideThickness();

  const T kB = 8.617333e-5;
  const T temperature = 1273.15;
  const T B = 386. * std::exp(-0.78 / (kB * temperature));
  const T BoA = 9.70e7 * std::exp(-2.05 / (kB * temperature));
  const T A = B / BoA;
  const T expected =
      (std::sqrt(A * A + 4. * B * 0.1) - A) / 2.;

  VC_TEST_ASSERT_ISCLOSE(thickness, expected, 1.e-12)
  VC_TEST_ASSERT(thickness > 0.)
}

void testOxidationCallbackCreatesNativeOxide() {
  auto domain = ps::Domain<T, D>::New();
  ps::MakePlane<T, D>(domain, 0.1, 1.0, 1.0, 0., false, ps::Material::Si)
      .apply();
  VC_TEST_ASSERT(domain->getNumberOfLevelSets() == 1)

  auto model = ps::SmartPointer<ps::Oxidation<T, D>>::New();
  model->setTemperature(1000.);
  model->setTime(0.);
  model->setOxidant(ps::OxidantType::Dry);
  model->setInitialOxideThickness(0.1);

  ps::Process<T, D>(domain, model, T(0)).apply();

  VC_TEST_ASSERT(domain->getNumberOfLevelSets() == 2)
  VC_TEST_ASSERT(domain->getMaterialMap()->getMaterialAtIdx(0) ==
                 ps::Material::Si)
  VC_TEST_ASSERT(domain->getMaterialMap()->getMaterialAtIdx(1) ==
                 ps::Material::SiO2)
}

int main() {
  testDealGroveEstimateWet1000C();
  testOxidationCallbackCreatesNativeOxide();
}
