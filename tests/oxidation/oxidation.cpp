#include <geometries/psMakePlane.hpp>
#include <models/psOxidation.hpp>
#include <process/psProcess.hpp>
#include <psDomain.hpp>

#include <lsGeometricAdvect.hpp>
#include <lsMakeGeometry.hpp>
#include <lsToSurfaceMesh.hpp>

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
  model->setOxidant(ps::OxidantType::WET);
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
  model->setOxidant(ps::OxidantType::DRY);
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
  model->setOxidant(ps::OxidantType::WET);
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
  model->setOxidant(ps::OxidantType::DRY);
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
  model->setOxidant(ps::OxidantType::DRY);
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
    m->setOxidant(ps::OxidantType::WET);
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

// Si consumed by a flat Si(100) film: the mean height of the Si surface below
// its start.  The wafer normal is the last axis in 2-D and 3-D alike.
template <int Dim> T flatFilmConsumption() {
  auto domain = ps::Domain<T, Dim>::New();
  ps::MakePlane<T, Dim>(domain, 0.01, 0.04, 0.04, 0., false, ps::Material::Si)
      .apply();
  auto model = ps::SmartPointer<ps::Oxidation<T, Dim>>::New();
  model->setTemperature(1000.);
  model->setTime(0.01);
  model->setOxidant(ps::OxidantType::WET);
  model->setOrientation(ps::SiliconOrientation::Si100);
  model->setInitialOxideThickness(0.03);
  ps::Process<T, Dim>(domain, model, T(0)).apply();

  auto mesh = viennals::SmartPointer<viennals::Mesh<T>>::New();
  viennals::ToSurfaceMesh<T, Dim>(domain->getLevelSets().front(), mesh).apply();
  T height = 0.;
  for (const auto &node : mesh->getNodes())
    height += node[Dim - 1];
  return -height / static_cast<T>(mesh->getNodes().size());
}

// A flat (100) film oxidises alike in 2-D and 3-D.  With the crystal axis
// fixed to y, the 3-D (z-up) film was rated as a (110)-like face (x1.45) and
// consumed ~36 % more Si.
void testFlatFilm3DMatches2D() {
  const T consumed2D = flatFilmConsumption<2>();
  const T consumed3D = flatFilmConsumption<3>();
  VC_TEST_ASSERT(consumed2D > T(0))
  VC_TEST_ASSERT_ISCLOSE(consumed3D / consumed2D, T(1), T(0.03))
}

// Mean height of the Si surface and of the oxide surface of a flat film.
std::array<T, 2> flatSurfaces(ps::SmartPointer<ps::Domain<T, D>> domain) {
  auto mean = [](auto levelSet) {
    auto mesh = viennals::SmartPointer<viennals::Mesh<T>>::New();
    viennals::ToSurfaceMesh<T, D>(levelSet, mesh).apply();
    T height = 0.;
    for (const auto &node : mesh->getNodes())
      height += node[D - 1];
    return height / static_cast<T>(mesh->getNodes().size());
  };
  return {mean(domain->getLevelSets().front()),
          mean(domain->getLevelSets().back())};
}

// Bare silicon on a grid too coarse for a native oxide (5 nm): the oxide
// grows by the 1-D Deal-Grove law until it spans three cells (15 nm, 0.36 hr
// dry at 1000 C), then the 2-D solve continues.  Both stages follow
// Deal-Grove from t = 0 (A = B/k + B/h, h = 100 um/hr the default gas
// transfer), and the Si recedes by 1/2.27 of the oxide.
void testStartupFromBareSilicon() {
  const T kT = 8.617333e-5 * 1273.15;
  const T B = 772. * std::exp(-1.23 / kT);
  const T A = B / (3.71e6 * std::exp(-2.00 / kT)) + B / 100.;
  for (const T time : {T(0.2), T(0.6)}) { // inside / beyond the start-up
    auto domain = ps::Domain<T, D>::New();
    ps::MakePlane<T, D>(domain, 0.005, 0.04, 0.04, 0., false, ps::Material::Si)
        .apply();
    auto model = ps::SmartPointer<ps::Oxidation<T, D>>::New();
    model->setTemperature(1000.);
    model->setTime(time);
    model->setOxidant(ps::OxidantType::DRY);
    model->setOrientation(ps::SiliconOrientation::Si100);
    model->setInitialOxideThickness(0.);
    ps::Process<T, D>(domain, model, T(0)).apply();

    VC_TEST_ASSERT(domain->getNumberOfLevelSets() == 2)
    const auto [si, top] = flatSurfaces(domain);
    const T expected = (std::sqrt(A * A + 4. * B * time) - A) / 2.;
    VC_TEST_ASSERT_ISCLOSE(top - si, expected, 0.01 * expected)
    VC_TEST_ASSERT_ISCLOSE(-si, expected / 2.27, 0.01 * expected / 2.27)
  }
}

// Si3N4 resting on bare silicon (padless LOCOS), 5 nm grid, dry 1000 C:
// the open Si grows by Deal-Grove from t = 0 while the masked Si is not
// touched, inside the 1-D start-up (0.2 hr) and after the 2-D solve took over
// (0.5 hr; the start-up ends at 0.36 hr).
void testStartupUnderMask() {
  namespace ls = viennals;
  constexpr T gridDelta = 0.005;
  double bounds[2 * D] = {-0.1, 0.1, -0.2, 0.3};
  ls::Domain<T, D>::BoundaryType bc[D] = {
      ls::Domain<T, D>::BoundaryType::REFLECTIVE_BOUNDARY,
      ls::Domain<T, D>::BoundaryType::INFINITE_BOUNDARY};
  const T kT = 8.617333e-5 * 1273.15;
  const T B = 772. * std::exp(-1.23 / kT);
  const T A = B / (3.71e6 * std::exp(-2.00 / kT)) + B / 100.;
  for (const T time : {T(0.2), T(0.5)}) {
    auto siLS = ls::Domain<T, D>::New(bounds, bc, gridDelta);
    ls::MakeGeometry<T, D>(siLS,
                           ls::Plane<T, D>::New(ls::VectorType<T, D>{0., 0.},
                                                ls::VectorType<T, D>{0., 1.}))
        .apply();
    auto maskLS = ls::Domain<T, D>::New(bounds, bc, gridDelta);
    ls::MakeGeometry<T, D> maskGeom(
        maskLS, ls::Box<T, D>::New(ls::VectorType<T, D>{-0.2, -1e-6},
                                   ls::VectorType<T, D>{0., 0.1}));
    maskGeom.setIgnoreBoundaryConditions(std::array<bool, D>{false, true});
    maskGeom.apply();
    auto domain = ps::Domain<T, D>::New();
    domain->insertNextLevelSetAsMaterial(siLS, ps::Material::Si, false);
    domain->insertNextLevelSetAsMaterial(maskLS, ps::Material::Si3N4, false);

    auto model = ps::SmartPointer<ps::Oxidation<T, D>>::New();
    model->setTemperature(1000.);
    model->setTime(time);
    model->setOxidant(ps::OxidantType::DRY);
    model->setOrientation(ps::SiliconOrientation::Si100);
    model->setInitialOxideThickness(0.);
    model->setMaxGridPoints(200000);
    ps::Process<T, D>(domain, model, T(0)).apply();
    VC_TEST_ASSERT(domain->getNumberOfLevelSets() == 3)

    // mean height of a level set's surface over x in [x0, x1]
    auto height = [](auto levelSet, T x0, T x1) {
      auto mesh = viennals::SmartPointer<viennals::Mesh<T>>::New();
      viennals::ToSurfaceMesh<T, D>(levelSet, mesh).apply();
      T sum = 0.;
      int count = 0;
      for (const auto &node : mesh->getNodes())
        if (node[0] >= x0 && node[0] <= x1 && node[1] < T(0.05)) {
          sum += node[1];
          ++count;
        }
      return count ? sum / count : T(1e9);
    };
    auto si = domain->getLevelSets()[0];
    auto oxide = domain->getLevelSets()[2];
    const T expected = (std::sqrt(A * A + 4. * B * time) - A) / 2.;
    const T open = height(oxide, 0.05, 0.1) - height(si, 0.05, 0.1);
    VC_TEST_ASSERT_ISCLOSE(open, expected, 0.02 * expected)
    // far under the mask: the silicon is untouched
    VC_TEST_ASSERT_ISCLOSE(height(si, -0.1, -0.05), T(0), T(1e-4))
  }
}

int main() {
  testDealGroveEstimateWet1000C();
  testDealGroveEstimateDryHighT();
  testDealGroveEstimateDryLowT();
  testOrientationRatios();
  testOxidationCallbackCreatesNativeOxide();
  testLocosOxidationPreservesLayers();
  testFlatFilm3DMatches2D();
  testStartupFromBareSilicon();
  testStartupUnderMask();
}
