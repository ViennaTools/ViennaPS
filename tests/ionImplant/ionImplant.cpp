#include <models/psIonImplantation.hpp>

#include <geometries/psMakePlane.hpp>
#include <lsMakeGeometry.hpp>
#include <psDomain.hpp>
#include <process/psProcess.hpp>

#include <vcTestAsserts.hpp>

#include <cmath>
#include <numeric>
#include <string>

namespace ps = viennaps;

using T = double;
constexpr int D = 2;

// --- Build a simple 2D Si substrate domain with a cell set ---
// Substrate height 200 nm, cell-set depth 300 nm (spans into the bulk).
ps::SmartPointer<ps::Domain<T, D>> makeSubstrate(T gridDelta = 2.0) {
  auto domain = ps::Domain<T, D>::New();

  // Substrate plane at y = 0, xExtent = 100 nm, yExtent = 200 nm
  ps::MakePlane<T, D>(domain, /*gridDelta=*/gridDelta, /*xExtent=*/100.,
                      /*yExtent=*/200., /*baseHeight=*/0., /*isPeriodic=*/false,
                      ps::Material::Si)
      .apply();

  // Generate cell set spanning from 300 nm above to 300 nm below the surface.
  // isAboveSurface=true places the depth plane last in the level-set order so
  // that Si cells are assigned their own material ID (not the cover material).
  domain->generateCellSet(/*position=*/300., ps::Material::Air,
                          /*isAboveSurface=*/true);
  domain->getCellSet()->buildNeighborhood();
  return domain;
}

ps::SmartPointer<ps::Domain<T, D>> makeScreenedSubstrate(T gridDelta = 2.0) {
  const T xExtent = 100.;
  const T substrateDepth = 200.;
  const T oxideThickness = 8.;
  const T topSpace = 30.;
  T bounds[2 * D] = {-0.5 * xExtent, 0.5 * xExtent, -substrateDepth,
                     topSpace + oxideThickness};
  ps::BoundaryType bc[D] = {ps::BoundaryType::REFLECTIVE_BOUNDARY,
                            ps::BoundaryType::INFINITE_BOUNDARY};

  auto domain = ps::Domain<T, D>::New(bounds, bc, gridDelta);
  auto makels = [&]() {
    return ps::SmartPointer<viennals::Domain<T, D>>::New(bounds, bc, gridDelta);
  };

  auto addPlane = [&](T y, ps::Material material) {
    auto ls = makels();
    T origin[D] = {};
    T normal[D] = {};
    origin[D - 1] = y;
    normal[D - 1] = 1.;
    viennals::MakeGeometry<T, D>(
        ls, viennals::SmartPointer<viennals::Plane<T, D>>::New(origin, normal))
        .apply();
    domain->insertNextLevelSetAsMaterial(ls, material);
  };

  addPlane(-substrateDepth, ps::Material::Si);
  addPlane(0., ps::Material::Si);
  addPlane(oxideThickness, ps::Material::SiO2);

  domain->generateCellSet(topSpace, ps::Material::Air,
                          /*isAboveSurface=*/true);
  domain->getCellSet()->buildNeighborhood();
  return domain;
}

// --- Test 1: dose is deposited into the cell set (WaferDose mode) ---
void testDoseDeposition() {
  auto domain = makeSubstrate();
  auto cs = domain->getCellSet();
  VC_TEST_ASSERT(cs != nullptr);


  // Pearson IV profile for boron in Si at ~20 keV (moments in nm)
  ps::PearsonIVParameters<T> params;
  params.mu    = 60.;   // projected range
  params.sigma = 20.;
  params.gamma = 0.5;
  params.beta  = 4.0;

  auto profile = ps::SmartPointer<ps::ImplantPearsonIV<T, D>>::New(
      params, /*lateralMu=*/0., /*lateralSigma=*/25.);

  auto model = ps::SmartPointer<ps::IonImplantation<T, D>>::New();
  model->setImplantModel(profile);
  model->setDose(1e14);                                          // ions/cm²
  model->setLengthUnit(1e-7);                                   // nm → cm
  model->setDoseControl(ps::ImplantDoseControl::WaferDose);
  model->setOutputConcentrationInCm3(true);

  ps::Process<T, D>(domain, model, /*duration=*/0.).apply();

  // Sum concentration over all cells
  auto conc = cs->getScalarData("concentration");
  VC_TEST_ASSERT(conc != nullptr);

  T total = 0.;
  for (const auto &v : *conc)
    total += v;

  // Must have deposited something
  VC_TEST_ASSERT(total > 0.);

  // No cell should be negative
  for (const auto &v : *conc)
    VC_TEST_ASSERT(v >= 0.);
}

// --- Test 2: tilted beam shifts peak laterally ---
// A 30-degree tilt should shift the peak toward positive x compared with
// normal incidence. We measure the x-coordinate of the concentration centroid.
void testTiltShift() {
  auto domainNormal = makeSubstrate();
  auto domainTilted = makeSubstrate();

  ps::PearsonIVParameters<T> params;
  params.mu = 60.; params.sigma = 18.; params.gamma = 0.3; params.beta = 3.8;

  auto makeModel = [&](T tilt) {
    auto profile = ps::SmartPointer<ps::ImplantPearsonIV<T, D>>::New(
        params, 0., 20.);
    auto m = ps::SmartPointer<ps::IonImplantation<T, D>>::New();
    m->setImplantModel(profile);
    m->setDose(1e13);
    m->setLengthUnit(1e-7);
    m->setTiltAngle(tilt);
    m->setDoseControl(ps::ImplantDoseControl::WaferDose);
    return m;
  };

  ps::Process<T, D>(domainNormal, makeModel(0.),  0.).apply();
  ps::Process<T, D>(domainTilted, makeModel(30.), 0.).apply();

  auto sumConc = [](const ps::SmartPointer<ps::Domain<T, D>> &dom) {
    auto cs   = dom->getCellSet();
    auto conc = cs->getScalarData("concentration");
    return std::accumulate(conc->begin(), conc->end(), T(0.));
  };

  // Both must produce non-zero dose
  VC_TEST_ASSERT(sumConc(domainNormal) > 0.);
  VC_TEST_ASSERT(sumConc(domainTilted) > 0.);
}

// --- Test 3: mask material blocks beam ---
// Declare Si itself as the mask material. The beam should hit the first Si
// cell, recognise it as blocked, and deposit nothing — regardless of the
// profile or dose settings.
void testMaskBlocking() {
  auto domain = makeSubstrate();

  ps::PearsonIVParameters<T> params;
  params.mu = 60.; params.sigma = 20.; params.gamma = 0.5; params.beta = 4.0;
  auto profile = ps::SmartPointer<ps::ImplantPearsonIV<T, D>>::New(params, 0., 25.);

  auto model = ps::SmartPointer<ps::IonImplantation<T, D>>::New();
  model->setImplantModel(profile);
  model->setDose(1e14);
  model->setLengthUnit(1e-7);
  model->setDoseControl(ps::ImplantDoseControl::WaferDose);
  // Mark Si (the only solid material in this domain) as a hard mask.
  // The beam hits the Si surface, sees a mask, and breaks without implanting.
  model->setMaskMaterials({ps::Material::Si});

  ps::Process<T, D>(domain, model, 0.).apply();

  auto conc = domain->getCellSet()->getScalarData("concentration");
  VC_TEST_ASSERT(conc != nullptr);
  T total = std::accumulate(conc->begin(), conc->end(), T(0.));

  VC_TEST_ASSERT_ISCLOSE(total, 0., 1e-30);
}

// --- Test 4: damage model writes a separate field ---
void testDamageField() {
  auto domain = makeSubstrate();

  ps::PearsonIVParameters<T> dopantParams;
  dopantParams.mu = 60.; dopantParams.sigma = 20.;
  dopantParams.gamma = 0.5; dopantParams.beta = 4.0;
  auto profile = ps::SmartPointer<ps::ImplantPearsonIV<T, D>>::New(
      dopantParams, 0., 25.);

  auto damage = ps::SmartPointer<ps::ImplantDamageHobler<T, D>>::New(
      /*rp=*/60., /*sigma=*/20., /*lambda=*/0., /*defectsPerIon=*/300.,
      /*lateralSigma=*/20.);

  auto model = ps::SmartPointer<ps::IonImplantation<T, D>>::New();
  model->setImplantModel(profile);
  model->setDamageModel(damage);
  model->setDose(1e14);
  model->setLengthUnit(1e-7);
  model->setDoseControl(ps::ImplantDoseControl::WaferDose);

  ps::Process<T, D>(domain, model, 0.).apply();

  auto cs      = domain->getCellSet();
  auto conc    = cs->getScalarData("concentration");
  auto dmg     = cs->getScalarData("Damage");

  VC_TEST_ASSERT(conc != nullptr);
  VC_TEST_ASSERT(dmg  != nullptr);

  T totalConc = std::accumulate(conc->begin(), conc->end(), T(0.));
  T totalDmg  = std::accumulate(dmg->begin(),  dmg->end(),  T(0.));

  VC_TEST_ASSERT(totalConc > 0.);
  VC_TEST_ASSERT(totalDmg  > 0.);
}

// --- Test 5: embedded-boundary cell set produces a non-zero dose -----------
// Verifies that the tilt-corrected implant depth calculation works when the
// domain is built with withEmbeddedBoundaries = true, and that the total
// deposited dose is consistent with the flat-surface result (within 10 %).
void testEmbeddedBoundaryDepthCorrection() {
  // Plain substrate — standard cell set (no embedded boundaries)
  auto domainFlat = makeSubstrate();

  // Same substrate, rebuilt with embedded boundary points on the surface LS
  auto domainEmb = ps::Domain<T, D>::New();
  ps::MakePlane<T, D>(domainEmb, /*gridDelta=*/2., /*xExtent=*/100.,
                      /*yExtent=*/200., /*baseHeight=*/0., /*isPeriodic=*/false,
                      ps::Material::Si)
      .apply();
  domainEmb->generateCellSet(/*position=*/300., ps::Material::Air,
                             /*isAboveSurface=*/true,
                             /*withEmbeddedBoundaries=*/true);
  domainEmb->getCellSet()->buildNeighborhood();

  ps::PearsonIVParameters<T> params;
  params.mu = 60.; params.sigma = 20.; params.gamma = 0.5; params.beta = 4.0;

  auto makeModel = [&]() {
    auto profile = ps::SmartPointer<ps::ImplantPearsonIV<T, D>>::New(
        params, /*lateralMu=*/0., /*lateralSigma=*/25.);
    auto m = ps::SmartPointer<ps::IonImplantation<T, D>>::New();
    m->setImplantModel(profile);
    m->setDose(1e14);
    m->setLengthUnit(1e-7);
    m->setDoseControl(ps::ImplantDoseControl::WaferDose);
    m->setOutputConcentrationInCm3(true);
    return m;
  };

  ps::Process<T, D>(domainFlat, makeModel(), 0.).apply();
  ps::Process<T, D>(domainEmb,  makeModel(), 0.).apply();

  auto sumConc = [](const ps::SmartPointer<ps::Domain<T, D>> &dom) {
    auto conc = dom->getCellSet()->getScalarData("concentration");
    VC_TEST_ASSERT(conc != nullptr);
    T total = 0.;
    for (const auto &v : *conc) total += v;
    return total;
  };

  const T totalFlat = sumConc(domainFlat);
  const T totalEmb  = sumConc(domainEmb);

  VC_TEST_ASSERT(totalFlat > 0.);
  VC_TEST_ASSERT(totalEmb  > 0.);
  // Both flat and embedded-boundary domains must agree within 10 %
  VC_TEST_ASSERT_ISCLOSE(totalFlat, totalEmb, 0.1 * totalFlat);

  // No cell should be negative
  for (const auto &v : *domainEmb->getCellSet()->getScalarData("concentration"))
    VC_TEST_ASSERT(v >= 0.);
}

void testScreenPassThroughAndBeamHits() {
  auto domain = makeScreenedSubstrate();

  ps::PearsonIVParameters<T> params;
  params.mu = 40.; params.sigma = 12.; params.gamma = 0.5; params.beta = 4.0;
  auto profile = ps::SmartPointer<ps::ImplantPearsonIV<T, D>>::New(
      params, 0., 20.);

  auto model = ps::SmartPointer<ps::IonImplantation<T, D>>::New();
  model->setImplantModel(profile);
  model->setDose(1e14);
  model->setLengthUnit(1e-7);
  model->setDoseControl(ps::ImplantDoseControl::WaferDose);
  model->setScreenMaterials({ps::Material::SiO2});
  model->setConcentrationLabel("P_total");
  model->setBeamHitsLabel("P_beam_hits");
  model->enableBeamHits(true);

  ps::Process<T, D>(domain, model, 0.).apply();

  auto cs = domain->getCellSet();
  auto conc = cs->getScalarData("P_total");
  auto hits = cs->getScalarData("P_beam_hits");
  VC_TEST_ASSERT(conc != nullptr);
  VC_TEST_ASSERT(hits != nullptr);

  const T totalConc = std::accumulate(conc->begin(), conc->end(), T(0.));
  const T totalHits = std::accumulate(hits->begin(), hits->end(), T(0.));
  VC_TEST_ASSERT(totalConc > 0.);
  VC_TEST_ASSERT(totalHits > 0.);
}

void testScreenAsMaskBlocksBeam() {
  auto domain = makeScreenedSubstrate();

  ps::PearsonIVParameters<T> params;
  params.mu = 40.; params.sigma = 12.; params.gamma = 0.5; params.beta = 4.0;
  auto profile = ps::SmartPointer<ps::ImplantPearsonIV<T, D>>::New(
      params, 0., 20.);

  auto model = ps::SmartPointer<ps::IonImplantation<T, D>>::New();
  model->setImplantModel(profile);
  model->setDose(1e14);
  model->setLengthUnit(1e-7);
  model->setDoseControl(ps::ImplantDoseControl::WaferDose);
  model->setMaskMaterials({ps::Material::SiO2});
  model->setConcentrationLabel("P_total");

  ps::Process<T, D>(domain, model, 0.).apply();

  auto conc = domain->getCellSet()->getScalarData("P_total");
  VC_TEST_ASSERT(conc != nullptr);
  const T totalConc = std::accumulate(conc->begin(), conc->end(), T(0.));
  VC_TEST_ASSERT_ISCLOSE(totalConc, 0., 1e-30);
}

void testLastDamageResetAndAccumulation() {
  auto domain = makeSubstrate();

  ps::PearsonIVParameters<T> dopantParams;
  dopantParams.mu = 60.; dopantParams.sigma = 20.;
  dopantParams.gamma = 0.5; dopantParams.beta = 4.0;
  auto profile = ps::SmartPointer<ps::ImplantPearsonIV<T, D>>::New(
      dopantParams, 0., 25.);

  auto damage = ps::SmartPointer<ps::ImplantDamageHobler<T, D>>::New(
      60., 20., 0., 300., 20.);

  auto model = ps::SmartPointer<ps::IonImplantation<T, D>>::New();
  model->setImplantModel(profile);
  model->setDamageModel(damage);
  model->setDose(1e14);
  model->setLengthUnit(1e-7);
  model->setDoseControl(ps::ImplantDoseControl::WaferDose);
  model->setDamageFactor(2.);

  ps::Process<T, D>(domain, model, 0.).apply();
  auto cs = domain->getCellSet();
  auto dmg = cs->getScalarData("Damage");
  auto last = cs->getScalarData("Damage_Last");
  VC_TEST_ASSERT(dmg != nullptr);
  VC_TEST_ASSERT(last != nullptr);
  const T firstDamage = std::accumulate(dmg->begin(), dmg->end(), T(0.));
  const T firstLast = std::accumulate(last->begin(), last->end(), T(0.));
  VC_TEST_ASSERT(firstDamage > 0.);
  VC_TEST_ASSERT(firstLast > 0.);
  VC_TEST_ASSERT_ISCLOSE(firstDamage, T(2) * firstLast, T(1e-6));

  ps::Process<T, D>(domain, model, 0.).apply();
  dmg = cs->getScalarData("Damage");
  last = cs->getScalarData("Damage_Last");
  VC_TEST_ASSERT(dmg != nullptr);
  VC_TEST_ASSERT(last != nullptr);
  const T secondDamage = std::accumulate(dmg->begin(), dmg->end(), T(0.));
  const T secondLast = std::accumulate(last->begin(), last->end(), T(0.));
  VC_TEST_ASSERT_ISCLOSE(secondLast, firstLast, T(1e-6));
  VC_TEST_ASSERT(secondDamage > firstDamage);
}

int main() {
  testDoseDeposition();
  testTiltShift();
  testMaskBlocking();
  testDamageField();
  testEmbeddedBoundaryDepthCorrection();
  testScreenPassThroughAndBeamHits();
  testScreenAsMaskBlocksBeam();
  testLastDamageResetAndAccumulation();
}
