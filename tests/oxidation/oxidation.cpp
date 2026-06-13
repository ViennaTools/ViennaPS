#include <geometries/psMakePlane.hpp>
#include <models/psOxidation.hpp>
#include <process/psProcess.hpp>
#include <psDomain.hpp>

#include <lsGeometricAdvect.hpp>
#include <lsMakeGeometry.hpp>

#include <vcTestAsserts.hpp>

#include <array>
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
  const T expected = (std::sqrt(A * A + 4. * B * 0.1) - A) / 2.;

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

// Verify that psOxidation activates LOCOS physics when a Si3N4 layer is present
// and that all three level sets (Si, SiO2, Si3N4) are preserved after the step.
void testLocosOxidationPreservesLayers() {
  namespace ls = viennals;

  constexpr T gridDelta = 0.1;
  constexpr T xExtent = 1.5;
  constexpr T yMin = -0.5;
  constexpr T yMax = 1.5;
  constexpr T padOxide = 0.15;
  constexpr T maskThick = 0.2;

  double bounds[2 * D] = {-xExtent, xExtent, yMin, yMax};
  ls::Domain<T, D>::BoundaryType bc[D] = {
      ls::Domain<T, D>::BoundaryType::REFLECTIVE_BOUNDARY,
      ls::Domain<T, D>::BoundaryType::INFINITE_BOUNDARY};

  // Si substrate (plane at y = 0).
  auto siLS = ls::Domain<T, D>::New(bounds, bc, gridDelta);
  ls::MakeGeometry<T, D>(siLS,
                         ls::Plane<T, D>::New(ls::VectorType<T, D>{0., 0.},
                                              ls::VectorType<T, D>{0., 1.}))
      .apply();

  // Pad oxide (geometrically expanded from Si surface).
  auto oxLS = ls::Domain<T, D>::New(siLS);
  auto sphere =
      ls::SmartPointer<ls::SphereDistribution<viennahrle::CoordType, D>>::New(
          padOxide);
  ls::GeometricAdvect<T, D>(oxLS, sphere).apply();

  // Si3N4 mask covering x < 0, sitting on top of pad oxide.
  auto maskLS = ls::Domain<T, D>::New(bounds, bc, gridDelta);
  ls::MakeGeometry<T, D> maskGeom(
      maskLS,
      ls::Box<T, D>::New(ls::VectorType<T, D>{-xExtent, padOxide - 1e-6},
                         ls::VectorType<T, D>{0., padOxide + maskThick}));
  maskGeom.setIgnoreBoundaryConditions(std::array<bool, D>{false, true});
  maskGeom.apply();

  auto domain = ps::Domain<T, D>::New();
  domain->insertNextLevelSetAsMaterial(siLS, ps::Material::Si, false);
  domain->insertNextLevelSetAsMaterial(oxLS, ps::Material::SiO2, false);
  domain->insertNextLevelSetAsMaterial(maskLS, ps::Material::Si3N4, false);

  VC_TEST_ASSERT(domain->getNumberOfLevelSets() == 3)

  auto model = ps::SmartPointer<ps::Oxidation<T, D>>::New();
  model->setTemperature(1000.);
  model->setTime(0.02);
  model->setOxidant(ps::OxidantType::Wet);
  model->setTimeStep(0.02);
  model->setMaxGridPoints(200000);

  ps::Process<T, D>(domain, model, T(0)).apply();

  // All three level sets must survive the LOCOS step.
  VC_TEST_ASSERT(domain->getNumberOfLevelSets() == 3)
  VC_TEST_ASSERT(domain->getMaterialMap()->getMaterialAtIdx(0) ==
                 ps::Material::Si)
  VC_TEST_ASSERT(domain->getMaterialMap()->getMaterialAtIdx(1) ==
                 ps::Material::SiO2)
  VC_TEST_ASSERT(domain->getMaterialMap()->getMaterialAtIdx(2) ==
                 ps::Material::Si3N4)
}

void testDealGroveEstimateDryHighT() {
  // T = 1000 °C ≥ 950 °C → high-T dry regime: Ea(B/A) = 2.00 eV, BoA0 = 3.71e6
  auto model = ps::SmartPointer<ps::Oxidation<T, D>>::New();
  model->setTemperature(1000.);
  model->setTime(0.5);
  model->setOxidant(ps::OxidantType::Dry);
  model->setPressure(1.);
  model->setOrientation(ps::SiliconOrientation::Si100);
  const T thickness = model->estimatePlanarOxideThickness();

  const T kB = 8.617333e-5;
  const T T_K = 1273.15;
  const T B = 772. * std::exp(-1.23 / (kB * T_K));
  const T BoA = 3.71e6 * std::exp(-2.00 / (kB * T_K));
  const T A = B / BoA;
  const T expected = (std::sqrt(A * A + 4. * B * 0.5) - A) / 2.;

  VC_TEST_ASSERT_ISCLOSE(thickness, expected, 1.e-12)
  VC_TEST_ASSERT(thickness > 0.)
}

void testDealGroveEstimateDryLowT() {
  // T = 900 °C < 950 °C → low-T dry regime: Ea(B/A) = 2.30 eV, BoA0 = 3.46e7
  auto model = ps::SmartPointer<ps::Oxidation<T, D>>::New();
  model->setTemperature(900.);
  model->setTime(0.5);
  model->setOxidant(ps::OxidantType::Dry);
  model->setPressure(1.);
  model->setOrientation(ps::SiliconOrientation::Si100);
  const T thickness = model->estimatePlanarOxideThickness();

  const T kB = 8.617333e-5;
  const T T_K = 1173.15;
  const T B = 772. * std::exp(-1.23 / (kB * T_K));
  const T BoA = 3.46e7 * std::exp(-2.30 / (kB * T_K));
  const T A = B / BoA;
  const T expected = (std::sqrt(A * A + 4. * B * 0.5) - A) / 2.;

  VC_TEST_ASSERT_ISCLOSE(thickness, expected, 1.e-12)
  VC_TEST_ASSERT(thickness > 0.)
}

// Verify that the Si110 and Si111 B/A scale factors relative to Si100
// match the established 1 : 1.45 : 1.68 ladder.  At very short times the
// linear regime dominates and thickness ∝ B/A, so the ratio of thicknesses
// directly reflects the orientation-dependent rate constants in dealGroveRow().
void testOrientationRatios() {
  constexpr T t = 0.001; // hr — firmly in the linear regime

  auto thickness = [&](ps::SiliconOrientation orientation) {
    auto m = ps::SmartPointer<ps::Oxidation<T, D>>::New();
    m->setTemperature(1000.);
    m->setTime(t);
    m->setOxidant(ps::OxidantType::Wet);
    m->setPressure(1.);
    m->setOrientation(orientation);
    return m->estimatePlanarOxideThickness();
  };

  const T x100 = thickness(ps::SiliconOrientation::Si100);
  const T x110 = thickness(ps::SiliconOrientation::Si110);
  const T x111 = thickness(ps::SiliconOrientation::Si111);

  VC_TEST_ASSERT(x100 > T(0))
  VC_TEST_ASSERT(x110 > x100)
  VC_TEST_ASSERT(x111 > x110)
  VC_TEST_ASSERT_ISCLOSE(x110 / x100, T(1.45), T(0.02))
  VC_TEST_ASSERT_ISCLOSE(x111 / x100, T(1.68), T(0.02))
}

int main() {
  testDealGroveEstimateWet1000C();
  testDealGroveEstimateDryHighT();
  testDealGroveEstimateDryLowT();
  testOrientationRatios();
  testOxidationCallbackCreatesNativeOxide();
  testLocosOxidationPreservesLayers();
}
