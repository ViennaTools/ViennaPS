#include <models/psAnneal.hpp>

#include <geometries/psMakePlane.hpp>
#include <psDomain.hpp>
#include <process/psProcess.hpp>

#include <vcTestAsserts.hpp>

#include <cmath>
#include <numeric>

namespace ps = viennaps;

using T = double;
constexpr int D = 2;

static constexpr T kGridDelta = 2.0; // nm

// --- Build a simple 2D Si substrate with a generous Air cell-set above it.
// isAboveSurface=true ensures Si cells get material ID 10, Air cells get 2.
// The narrow-band level set gives only 1–2 rows of Si; the Air region above
// (~0 to 300 nm) provides plenty of cells for diffusion tests.
ps::SmartPointer<ps::Domain<T, D>> makeSubstrate() {
  auto domain = ps::Domain<T, D>::New();
  ps::MakePlane<T, D>(domain, kGridDelta, /*xExtent=*/100., /*yExtent=*/200.,
                      /*baseHeight=*/0., /*isPeriodic=*/false,
                      ps::Material::Si)
      .apply();
  domain->generateCellSet(/*position=*/300., ps::Material::Air,
                          /*isAboveSurface=*/true);
  domain->getCellSet()->buildNeighborhood();
  return domain;
}

// --- Test 1: diffusion in the Air region spreads a peaked concentration ---
// Places a spike at mid-height of the Air region, anneals, then verifies the
// peak decreased and the total concentration is conserved.
void testDiffusionSpreads() {
  auto domain = makeSubstrate();
  auto cs = domain->getCellSet();
  VC_TEST_ASSERT(cs != nullptr);

  cs->addScalarData("concentration", T(0));
  auto conc = cs->getScalarData("concentration");
  auto mats = cs->getScalarData("Material");

  const int airId = static_cast<int>(ps::Material::Air);

  // Locate the y-range of Air cells
  T minY = 1e30, maxY = -1e30;
  for (int i = 0; i < cs->getNumberOfCells(); ++i) {
    if (static_cast<int>((*mats)[i]) != airId)
      continue;
    T y = cs->getCellCenter(i)[D - 1];
    minY = std::min(minY, y);
    maxY = std::max(maxY, y);
  }
  VC_TEST_ASSERT(minY < maxY); // domain must have Air cells

  // Spike in a single row near the mid-height of the Air region
  const T midY = 0.5 * (minY + maxY);
  T peakBefore = 0.;
  for (int i = 0; i < cs->getNumberOfCells(); ++i) {
    if (static_cast<int>((*mats)[i]) != airId)
      continue;
    T y = cs->getCellCenter(i)[D - 1];
    if (std::fabs(y - midY) < 1.5 * kGridDelta) {
      (*conc)[i] = 1.0;
      peakBefore = 1.0;
    }
  }
  VC_TEST_ASSERT(peakBefore > 0.);

  T sumBefore = 0.;
  for (int i = 0; i < cs->getNumberOfCells(); ++i)
    if (static_cast<int>((*mats)[i]) == airId)
      sumBefore += (*conc)[i];

  auto model = ps::SmartPointer<ps::Anneal<T, D>>::New();
  model->setDiffusionCoefficient(100.0); // length²/s
  model->setDuration(1.0);              // seconds
  model->setMode(ps::AnnealMode::Explicit);
  model->setDiffusionMaterials({ps::Material::Air});
  model->setBlockingMaterials({ps::Material::Si});

  ps::Process<T, D>(domain, model, T(0)).apply();

  T peakAfter = 0.;
  T sumAfter = 0.;
  for (int i = 0; i < cs->getNumberOfCells(); ++i) {
    if (static_cast<int>((*mats)[i]) != airId)
      continue;
    peakAfter = std::max(peakAfter, (*conc)[i]);
    sumAfter += (*conc)[i];
  }

  VC_TEST_ASSERT(peakAfter < peakBefore);
  VC_TEST_ASSERT_ISCLOSE(sumBefore, sumAfter, 1e-4 * sumBefore);

  for (int i = 0; i < cs->getNumberOfCells(); ++i)
    VC_TEST_ASSERT((*conc)[i] >= 0.);
}

// --- Test 2: temperature schedule (ramp/soak/ramp) runs without error ---
void testTemperatureSchedule() {
  auto domain = makeSubstrate();
  auto cs = domain->getCellSet();
  VC_TEST_ASSERT(cs != nullptr);

  cs->addScalarData("concentration", T(0));
  auto conc = cs->getScalarData("concentration");
  auto mats = cs->getScalarData("Material");

  const int airId = static_cast<int>(ps::Material::Air);
  int nAir = 0;
  for (int i = 0; i < cs->getNumberOfCells(); ++i) {
    if (static_cast<int>((*mats)[i]) == airId) {
      (*conc)[i] = 1.0;
      ++nAir;
    }
  }
  VC_TEST_ASSERT(nAir > 0);

  auto model = ps::SmartPointer<ps::Anneal<T, D>>::New();
  // Ea = 0 → D(T) = D0 = 1 at any temperature
  model->setArrheniusParameters(1.0, 0.);
  // ramp 800→1200 K / soak at 1200 K / ramp 1200→800 K
  model->setTemperatureSchedule({2.0, 2.0, 2.0}, {800., 1200., 1200., 800.});
  model->setDiffusionMaterials({ps::Material::Air});
  model->setBlockingMaterials({ps::Material::Si});

  ps::Process<T, D>(domain, model, T(0)).apply();

  // All concentrations must remain non-negative
  for (int i = 0; i < cs->getNumberOfCells(); ++i)
    VC_TEST_ASSERT((*conc)[i] >= 0.);
}

// --- Test 3: solid activation writes a bounded active_concentration field ---
// With total_C = 1e22 and C_SS = 1e19, the active fraction ≈ C_SS * total / (C_SS + total).
void testSolidActivation() {
  auto domain = makeSubstrate();
  auto cs = domain->getCellSet();
  VC_TEST_ASSERT(cs != nullptr);

  cs->addScalarData("concentration", T(0));
  auto conc = cs->getScalarData("concentration");
  auto mats = cs->getScalarData("Material");

  const int airId = static_cast<int>(ps::Material::Air);
  for (int i = 0; i < cs->getNumberOfCells(); ++i)
    if (static_cast<int>((*mats)[i]) == airId)
      (*conc)[i] = 1e22;

  auto model = ps::SmartPointer<ps::Anneal<T, D>>::New();
  model->setDiffusionCoefficient(0.);  // no diffusion; only activation
  model->setDuration(1.0);
  model->enableSolidActivation(true);
  // C_SS(T) = 1e19 · exp(0) = 1e19 (temperature-independent with Ea = 0)
  model->setSolidSolubilityArrhenius(1e19, 0.);
  model->setDiffusionMaterials({ps::Material::Air});
  model->setBlockingMaterials({ps::Material::Si});

  ps::Process<T, D>(domain, model, T(0)).apply();

  auto active = cs->getScalarData("active_concentration");
  VC_TEST_ASSERT(active != nullptr);

  for (int i = 0; i < cs->getNumberOfCells(); ++i) {
    if (static_cast<int>((*mats)[i]) != airId)
      continue;
    const T total = (*conc)[i];
    const T act   = (*active)[i];
    VC_TEST_ASSERT(act >= 0.);
    VC_TEST_ASSERT(act <= total + 1e-10);
    // active = C_SS · C_total / (C_SS + C_total)
    const T expectedActive = 1e19 * 1e22 / (1e19 + 1e22);
    VC_TEST_ASSERT_ISCLOSE(act, expectedActive, 1e16);
  }
}

int main() {
  testDiffusionSpreads();
  testTemperatureSchedule();
  testSolidActivation();
}
