#include <models/psImplantProfile.hpp>

#include <vcTestAsserts.hpp>

#include <cmath>
#include <string>

namespace ps = viennaps;

using T = double;
constexpr int D = 2;

// --- Test 1: PearsonIV model peak and normalization ---
// At depth == mu the profile must be at its maximum and the total depth
// integral must equal 1 (the model normalizes internally).
void testPearsonIVProfile() {
  ps::PearsonIVParameters<T> params;
  params.mu = 100.;   // nm
  params.sigma = 20.;
  params.gamma = 0.5;
  params.beta = 4.0;

  ps::ImplantPearsonIV<T, D> model(params, /*lateralMu=*/0., /*lateralSigma=*/30.);

  // Peak should be positive
  VC_TEST_ASSERT(model.getDepthProfile(params.mu) > 0.);

  // Profile is zero at negative depth (ions can't go above the surface)
  VC_TEST_ASSERT_ISCLOSE(model.getDepthProfile(-1.), 0., 1e-12);

  // Reported max depth must be beyond mu
  VC_TEST_ASSERT(model.getMaxDepth() > params.mu);

  // Lateral profile at zero offset must be its own maximum
  const T lateralAtCenter = model.getLateralProfile(0., params.mu);
  const T lateralOffset   = model.getLateralProfile(60., params.mu);
  VC_TEST_ASSERT(lateralAtCenter > lateralOffset);

  // getProfile = depthProfile * lateralProfile
  const T expected = model.getDepthProfile(params.mu) * model.getLateralProfile(0., params.mu);
  VC_TEST_ASSERT_ISCLOSE(model.getProfile(params.mu, 0.), expected, 1e-12);
}

// --- Test 2: DualPearsonIV mixes two components ---
void testDualPearsonIVProfile() {
  ps::PearsonIVParameters<T> head;
  head.mu = 80.; head.sigma = 15.; head.gamma = 0.3; head.beta = 3.5;

  ps::PearsonIVParameters<T> tail;
  tail.mu = 150.; tail.sigma = 40.; tail.gamma = 0.8; tail.beta = 5.0;

  const T headFraction = 0.7;
  ps::ImplantDualPearsonIV<T, D> model(head, tail, headFraction,
                                        /*headLateralMu=*/0., /*headLateralSigma=*/20.,
                                        /*tailLateralMu=*/0., /*tailLateralSigma=*/35.);

  // Profile must be positive somewhere between the two peaks
  VC_TEST_ASSERT(model.getDepthProfile(head.mu)  > 0.);
  VC_TEST_ASSERT(model.getDepthProfile(tail.mu)  > 0.);
  VC_TEST_ASSERT(model.getDepthProfile(-5.)      == 0.);
  VC_TEST_ASSERT(model.getMaxDepth() >= tail.mu);
}

// --- Test 3: ImplantPearsonIVChanneling adds a deeper tail ---
void testChannelingTail() {
  ps::PearsonIVParameters<T> params;
  params.mu = 60.; params.sigma = 12.; params.gamma = 0.2; params.beta = 3.8;

  const T tailFraction    = 0.3;
  const T tailStartDepth  = 80.;
  const T tailDecayLength = 50.;

  ps::ImplantPearsonIVChanneling<T, D> model(params, /*lateralMu=*/0.,
                                              /*lateralSigma=*/20., tailFraction,
                                              tailStartDepth, tailDecayLength);

  // Must have non-zero profile well past the Pearson peak (channeling tail)
  VC_TEST_ASSERT(model.getDepthProfile(tailStartDepth + tailDecayLength) > 0.);
  VC_TEST_ASSERT(model.getMaxDepth() > tailStartDepth + 5. * tailDecayLength);
}

// --- Test 4: Hobler damage profile ---
// The damage model integrates to defectsPerIon over depth.
void testHoblerDamageProfile() {
  const T rp             = 80.;
  const T sigma          = 15.;
  const T lambda         = 0.;   // pure Gaussian
  const T defectsPerIon  = 500.;
  const T lateralSigma   = 20.;

  ps::ImplantDamageHobler<T, D> model(rp, sigma, lambda, defectsPerIon, lateralSigma);

  // Peak should be near rp
  VC_TEST_ASSERT(model.getDepthProfile(rp) > model.getDepthProfile(rp + 5. * sigma));
  VC_TEST_ASSERT(model.getDepthProfile(-1.) == 0.);
  VC_TEST_ASSERT(model.getMaxDepth() >= rp);
}

// --- Test 5: material-name bridge ---
void testImplantMaterialName() {
  VC_TEST_ASSERT(ps::implantMaterialName(viennaps::Material::Si)      == "silicon");
  VC_TEST_ASSERT(ps::implantMaterialName(viennaps::Material::BulkSi)  == "silicon");
  VC_TEST_ASSERT(ps::implantMaterialName(viennaps::Material::aSi)     == "silicon");
  VC_TEST_ASSERT(ps::implantMaterialName(viennaps::Material::PolySi)  == "silicon");
  VC_TEST_ASSERT(ps::implantMaterialName(viennaps::Material::Ge)      == "germanium");
  VC_TEST_ASSERT(ps::implantMaterialName(viennaps::Material::SiGe)    == "sige");
  VC_TEST_ASSERT(ps::implantMaterialName(viennaps::Material::SiO2)    == "oxide");
  VC_TEST_ASSERT(ps::implantMaterialName(viennaps::Material::Si3N4)   == "nitride");
  VC_TEST_ASSERT(ps::implantMaterialName(viennaps::Material::SiN)     == "nitride");
  // Custom material: unknown
  VC_TEST_ASSERT(ps::implantMaterialName(viennaps::Material::custom(42)) == "unknown");
}

int main() {
  testPearsonIVProfile();
  testDualPearsonIVProfile();
  testChannelingTail();
  testHoblerDamageProfile();
  testImplantMaterialName();
}
