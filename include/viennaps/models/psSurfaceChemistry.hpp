#pragma once

#include <rayParticle.hpp>
#include <rayReflection.hpp>

#include "../process/psProcessModel.hpp"
#include "../process/psSurfaceModel.hpp"
#include "../process/psVelocityField.hpp"
#include "../psConstants.hpp"
#include "psIonModelUtil.hpp"
#include "../psUnits.hpp"

#include "../materials/psMaterialMap.hpp"
#include <vcKDTree.hpp>
#include "../materials/psMaterialValueMap.hpp"

#include <cmath>
#include <optional>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <limits>
#include <vector>

namespace viennaps {

using namespace viennacore;

// Generic, data-driven CVD surface chemistry.
//
// One class covers every mechanism. The chemistry enters as data (a
// ChemicalMechanism), not as a subclass, because all mechanisms perform the same
// two operations at each surface point:
//
//   solve    sum_j nu_ij * r_j(theta, Gamma) = 0     for every adsorbed species i
//   return   v_n = (1/rho_S) * sum_j n_S,j * r_j
//
// with mass-action rates
//
//   r_j = k_j * prod_g Gamma_g^n_gj * theta_free^n_*j * prod_i theta_i^n_ij
//
// The exponents come from the left-hand side of each reaction. The Jacobian is
// built from those same exponents by the product rule, including the chain term
// from theta_free = 1 - sum_i theta_i, so no symbolic algebra is needed at run
// time. Derivatives multiply the remaining factors instead of dividing, so a
// coverage of zero is safe.
//
// Sticking convention: an adsorption reaction's constant IS the sticking
// coefficient. It is applied once in the rate law here, and once in the
// particle's re-emission during transport. The particle records the raw incident
// flux. Applying it anywhere else counts it twice.
//
// Units: Gamma in 1e15 /cm^2 /s, rho_S in 1e22 /cm^3, so r/rho is nm/s directly.

template <typename NumericType> struct ChemicalMechanism {

  struct Factor {
    int index = 0;    // into coverages, or into the gas species list
    int exponent = 1; // left-hand-side coefficient
  };

  // k = prefactor * T^beta * exp(-Ea / kB T). Reactions and the sticking of a
  // traced species both carry one of these, per material.
  struct RateConstant {
    NumericType prefactor = 0.;
    NumericType Ea = 0.;
    NumericType beta = 0.;
  };

  struct Reaction {
    // k = prefactor * T^beta * exp(-Ea / kB T); for an adsorption step the
    // prefactor is the sticking coefficient s0. beta = 0 is plain Arrhenius.
    // beta = -1/2 arises when a sticking coefficient is derived from an
    // adsorption rate constant, s = 4 k_ads / vbar, since vbar ~ sqrt(T).
    NumericType prefactor = 0.;
    NumericType Ea = 0.;   // eV
    NumericType beta = 0.; // temperature exponent
    bool isAdsorption = false;

    std::vector<Factor> gasFactors;
    std::vector<Factor> coverageFactors;
    std::vector<int> freeSiteExponent; // one entry per site type

    std::string equation;        // the reaction as written, for reporting
    std::vector<NumericType> nu; // one entry per coverage
    NumericType solidAtoms = 0.; // solid atoms produced per event
    // Which solid phase those atoms belong to. A mechanism that deposits a
    // polymer while etching a substrate has two, and their densities differ,
    // so the same atom count moves the surface by different amounts.
    int solidIndex = 0;

    // The rate constant on a given material. A chemistry can differ between
    // materials in more than its overall rate: a different sticking, a
    // different barrier, or both, and a prefactor of zero means the step does
    // not occur there at all. The default applies to any material not named,
    // and is the reaction's own constant unless set otherwise.
    MaterialValueMap<RateConstant> materialConstant;
  };

  // An ion source. The energy and angle dependence of an ion-driven step lives
  // here and in the particle, never in the rate law: the particle deposits a
  // YIELD-WEIGHTED flux, so the surface chemistry treats it like any other flux
  // and the solver needs no notion of ions at all.
  struct IonSource {
    NumericType meanEnergy = 100.;  // eV
    NumericType sigmaEnergy = 10.;  // eV
    NumericType exponent = 200.;    // source distribution power
    NumericType inflectAngle = 89.; // degrees, for the energy loss on reflection
    NumericType n_l = 10.;
    NumericType minAngle = 80.;   // degrees, cone for the reflected direction
    NumericType thetaRMin = 70.;  // below this the ion is absorbed
    NumericType thetaRMax = 90.;  // above this it reflects fully
    bool present = false;
  };

  // One yield channel, produced by an ion-driven reaction. The particle
  // evaluates Y = A * max(sqrt(E) - sqrt(Eth), 0) * f(theta) at every hit and
  // adds Y * rayWeight to this channel's flux.
  struct IonYield {
    std::string label;      // the flux label this channel writes
    NumericType A = 1.;     // yield prefactor
    NumericType Eth = 0.;   // threshold energy, eV
    NumericType B = 0.;     // angular parameter of the sputtering form
    bool enhanced = false;  // true: the ion-enhanced angular form
    int gasIndex = 0;       // the gas species this channel writes into
    // per-material A and Eth, for selectivity to a mask or a stop layer
    MaterialValueMap<NumericType> materialA;
    MaterialValueMap<NumericType> materialEth;
  };

  struct GasSpecies {
    std::string label;         // flux label; empty when not traced
    NumericType sourceFlux = 0.;
    bool traced = false;
    // true when this "species" is an ion yield channel. It is traced, because
    // the chemistry reads its flux, but the ION writes it: it must not also get
    // a neutral particle of its own, or the flux is written twice.
    bool isIonChannel = false;
    // sticking of the adsorption step that consumes this species
    NumericType s0 = 0.;
    NumericType stickingEa = 0.;
    NumericType stickingBeta = 0.; // temperature exponent, -1/2 when from k_ads
    int stickingFreeSiteExponent = 0;
    int stickingSite = 0; // site type whose free sites this adsorption consumes

    // The sticking on a given material, taken from the adsorption step that
    // consumes this species. A selective chemistry adsorbs differently on
    // different materials, and the particle's re-emission has to follow, or the
    // transport inside a feature is wrong even when the surface solve is right.
    MaterialValueMap<RateConstant> stickingConstant;
  };

  // One solid phase of the mechanism: what a reaction deposits or removes.
  struct SolidPhase {
    std::string name;
    NumericType rho = 1.;
    // The density may differ per material: a mask and a substrate are not
    // equally dense, so the same removal rate moves them at different speeds.
    MaterialValueMap<NumericType> rhoMaterial;
    bool materialDependent = false;
  };

  std::string name;          // the mechanism's name, for reporting
  NumericType temperature = 300.;
  NumericType rhoSolid = 1.; // the density of solid 0 when none was declared
  std::vector<SolidPhase> solids;
  NumericType siteDensity = 0.; // optional; unused by the steady-state solve

  int numSiteTypes = 1;
  bool materialDependent = false; // true once any per-material factor is set
  std::vector<int> coverageSite;  // site-type index of each coverage
  std::vector<std::string> coverageNames;
  std::vector<NumericType> diffusionCoefficients; // per coverage, 0 disables
  std::vector<GasSpecies> gas;
  std::vector<Reaction> reactions;
  IonSource ionSource;
  std::vector<IonYield> ionYields; // one per ion-driven reaction

  // --- builder interface (used from Python) -----------------------------
  void setSiteTypeCount(int n) { numSiteTypes = std::max(1, n); }

  int addCoverage(const std::string &name, NumericType diffusion = 0.,
                  int site = 0) {
    coverageNames.push_back(name);
    diffusionCoefficients.push_back(diffusion);
    coverageSite.push_back(site);
    for (auto &r : reactions)
      r.nu.resize(coverageNames.size(), 0.);
    return static_cast<int>(coverageNames.size()) - 1;
  }

  int addGasSpecies(const std::string &label, NumericType sourceFlux,
                    bool traced) {
    GasSpecies g;
    g.label = traced ? label : std::string();
    g.sourceFlux = sourceFlux;
    g.traced = traced;
    gas.push_back(g);
    return static_cast<int>(gas.size()) - 1;
  }

  void setSticking(int gasIndex, NumericType s0, NumericType Ea,
                   int freeSiteExponent, NumericType beta = 0., int site = 0) {
    auto &g = gas.at(gasIndex);
    g.s0 = s0;
    g.stickingEa = Ea;
    g.stickingBeta = beta;
    g.stickingFreeSiteExponent = freeSiteExponent;
    g.stickingSite = site;
    // unless a material says otherwise, every material sticks like this
    g.stickingConstant =
        MaterialValueMap<RateConstant>::fromDefault({s0, Ea, beta});
  }

  // The sticking this species takes on a given material, mirroring the
  // per-material constant of the adsorption step that consumes it.
  void setStickingMaterialConstant(int gasIndex, Material material,
                                   NumericType s0, NumericType Ea,
                                   NumericType beta = 0.) {
    gas.at(gasIndex).stickingConstant.set(material, {s0, Ea, beta});
    materialDependent = true;
  }

  void setStickingMaterialConstantDefault(int gasIndex, NumericType s0,
                                          NumericType Ea,
                                          NumericType beta = 0.) {
    gas.at(gasIndex).stickingConstant.setDefault({s0, Ea, beta});
    materialDependent = true;
  }

  // Sticking of a traced species, evaluated for one material at the mechanism
  // temperature.
  NumericType stickingOf(int gasIndex, Material material) const {
    const auto &c = gas.at(gasIndex).stickingConstant.get(material);
    return arrhenius(c.prefactor, c.Ea, c.beta);
  }

  // This species' sticking evaluated once per material, so a particle does a
  // lookup per hit rather than an exponential.
  MaterialValueMap<NumericType> stickingTable(int gasIndex) const {
    const auto &g = gas.at(gasIndex);
    const auto &d = g.stickingConstant.getDefault();
    auto out = MaterialValueMap<NumericType>::fromDefault(
        arrhenius(d.prefactor, d.Ea, d.beta));
#define PS_EVAL_STICKING(id, sym, cat, dens, cond, color)                      \
  if (g.stickingConstant.has(BuiltInMaterial::sym)) {                          \
    const auto &c = g.stickingConstant.get(BuiltInMaterial::sym);              \
    out.set(Material(BuiltInMaterial::sym),                                    \
            arrhenius(c.prefactor, c.Ea, c.beta));                             \
  }
    BUILTIN_MATERIAL_LIST(PS_EVAL_STICKING)
#undef PS_EVAL_STICKING
    return out;
  }

  int addReaction(NumericType prefactor, NumericType Ea, bool isAdsorption,
                  const std::vector<int> &freeSiteExponent,
                  const std::vector<NumericType> &nu, NumericType solidAtoms,
                  NumericType beta = 0.) {
    Reaction r;
    r.prefactor = prefactor;
    r.Ea = Ea;
    r.beta = beta;
    r.isAdsorption = isAdsorption;
    r.freeSiteExponent = freeSiteExponent;
    r.freeSiteExponent.resize(numSiteTypes, 0);
    r.nu = nu;
    r.nu.resize(coverageNames.size(), 0.);
    r.solidAtoms = solidAtoms;
    // unless a material says otherwise, every material uses these constants
    r.materialConstant =
        MaterialValueMap<RateConstant>::fromDefault(
            {prefactor, Ea, beta});
    reactions.push_back(r);
    return static_cast<int>(reactions.size()) - 1;
  }

  // The incident flux of one gas species, by name. An atomic layer process
  // gates its steps this way: during the dose the radicals are not flowing,
  // during the plasma the precursor is not, and during a purge nothing is.
  // Returns false when the mechanism has no such species, so a driver that
  // misspells one is told rather than silently gating nothing.
  bool setSourceFlux(const std::string &label, NumericType flux) {
    for (auto &g : gas)
      if (g.label == label) {
        g.sourceFlux = flux;
        return true;
      }
    return false;
  }

  void addGasFactor(int reactionIndex, int gasIndex, int exponent) {
    reactions.at(reactionIndex).gasFactors.push_back({gasIndex, exponent});
  }

  // --- ions --------------------------------------------------------------
  void setIonSource(NumericType meanEnergy, NumericType sigmaEnergy,
                    NumericType exponent) {
    ionSource.meanEnergy = meanEnergy;
    ionSource.sigmaEnergy = sigmaEnergy;
    ionSource.exponent = exponent;
    ionSource.present = true;
  }

  void setIonReflection(NumericType inflectAngle, NumericType n_l,
                        NumericType minAngle, NumericType thetaRMin,
                        NumericType thetaRMax) {
    ionSource.inflectAngle = inflectAngle;
    ionSource.n_l = n_l;
    ionSource.minAngle = minAngle;
    ionSource.thetaRMin = thetaRMin;
    ionSource.thetaRMax = thetaRMax;
  }

  // Declare a yield channel. The returned index is the gas species the channel
  // writes into, so an ion-driven reaction takes it as an ordinary gas factor.
  int addIonYield(const std::string &label, NumericType A, NumericType Eth,
                  NumericType B = 0., bool enhanced = false,
                  NumericType sourceFlux = 1.) {
    IonYield y;
    y.label = label;
    y.A = A;
    y.Eth = Eth;
    y.B = B;
    y.enhanced = enhanced;
    y.materialA = MaterialValueMap<NumericType>::fromDefault(A);
    y.materialEth = MaterialValueMap<NumericType>::fromDefault(Eth);
    // the channel appears to the chemistry as a gas species carrying that
    // flux, scaled by the ion source flux. The ion writes it, so it is flagged
    // to keep the neutral particle loop from writing it as well.
    const int idx = addGasSpecies(label, sourceFlux, true);
    gas[idx].isIonChannel = true;
    y.gasIndex = idx;
    ionYields.push_back(y);
    return idx;
  }

  void setIonYieldMaterial(int yieldIndex, Material material, NumericType A,
                           NumericType Eth) {
    ionYields.at(yieldIndex).materialA.set(material, A);
    ionYields.at(yieldIndex).materialEth.set(material, Eth);
    materialDependent = true;
  }

  // The rate constant this reaction takes on a given material. Any of the
  // prefactor, the barrier and the temperature exponent may differ, so a
  // chemistry can vary between materials in more than its overall rate. A
  // prefactor of zero means the step does not occur on that material.
  void setMaterialConstant(int reactionIndex, Material material,
                           NumericType prefactor, NumericType Ea,
                           NumericType beta = 0.) {
    reactions.at(reactionIndex)
        .materialConstant.set(material, {prefactor, Ea, beta});
    materialDependent = true;
  }

  // Covers every material not named explicitly, so a selective chemistry sets
  // the default prefactor to zero and names the materials it proceeds on.
  void setMaterialConstantDefault(int reactionIndex, NumericType prefactor,
                                  NumericType Ea, NumericType beta = 0.) {
    reactions.at(reactionIndex)
        .materialConstant.setDefault({prefactor, Ea, beta});
    materialDependent = true;
  }

  // Rate constants evaluated for one material. Called once per material present
  // on the surface, so the exponential is not paid per surface point.
  std::vector<NumericType> rateConstantsFor(Material material) const {
    std::vector<NumericType> k;
    k.reserve(reactions.size());
    for (const auto &r : reactions) {
      const auto &c = r.materialConstant.get(material);
      k.push_back(arrhenius(c.prefactor, c.Ea, c.beta));
    }
    return k;
  }

  void addCoverageFactor(int reactionIndex, int coverageIndex, int exponent) {
    reactions.at(reactionIndex)
        .coverageFactors.push_back({coverageIndex, exponent});
  }

  // --- evaluation --------------------------------------------------------
  NumericType arrhenius(NumericType prefactor, NumericType Ea,
                        NumericType beta = 0.) const {
    NumericType k = prefactor * std::exp(-Ea / (constants::kB * temperature));
    if (beta != 0.)
      k *= std::pow(temperature, beta);
    return k;
  }

  std::vector<NumericType> rateConstants() const {
    std::vector<NumericType> k;
    k.reserve(reactions.size());
    for (const auto &r : reactions)
      k.push_back(arrhenius(r.prefactor, r.Ea, r.beta));
    return k;
  }

  NumericType stickingOf(int gasIndex) const {
    const auto &g = gas.at(gasIndex);
    return arrhenius(g.s0, g.stickingEa, g.stickingBeta);
  }

  static NumericType ipow(NumericType base, int e) {
    NumericType v = 1.;
    for (int i = 0; i < e; ++i)
      v *= base;
    return v;
  }

  NumericType gasPart(const Reaction &r, NumericType k,
                      const std::vector<NumericType> &gamma) const {
    NumericType v = k;
    for (const auto &f : r.gasFactors)
      v *= ipow(gamma[f.index], f.exponent);
    return v;
  }

  // Free-site fraction of each site type: theta_*t = 1 - sum_{i in t} theta_i.
  std::vector<NumericType>
  freeFractions(const std::vector<NumericType> &theta) const {
    std::vector<NumericType> occ(numSiteTypes, 0.);
    for (size_t i = 0; i < theta.size(); ++i)
      occ[coverageSite[i]] += theta[i];
    std::vector<NumericType> free(numSiteTypes);
    for (int t = 0; t < numSiteTypes; ++t)
      free[t] = std::max(NumericType(1.) - occ[t], NumericType(0.));
    return free;
  }

  // Product of theta_*t^n_t over the site types.
  NumericType freePart(const Reaction &r,
                       const std::vector<NumericType> &free) const {
    NumericType v = 1.;
    for (int t = 0; t < numSiteTypes; ++t)
      if (r.freeSiteExponent[t])
        v *= ipow(free[t], r.freeSiteExponent[t]);
    return v;
  }

  NumericType rate(const Reaction &r, NumericType k,
                   const std::vector<NumericType> &gamma,
                   const std::vector<NumericType> &theta,
                   const std::vector<NumericType> &free) const {
    NumericType v = gasPart(r, k, gamma) * freePart(r, free);
    for (const auto &f : r.coverageFactors)
      v *= ipow(theta[f.index], f.exponent);
    return v;
  }

  // d(rate)/d(theta_k), by the product rule over the stored exponents. theta_k
  // acts twice: as an explicit coverage factor, and through the free fraction of
  // its own site type coverageSite[kIdx].
  NumericType dRate(const Reaction &r, NumericType k,
                    const std::vector<NumericType> &gamma,
                    const std::vector<NumericType> &theta,
                    const std::vector<NumericType> &free, int kIdx) const {
    const NumericType base = gasPart(r, k, gamma);
    NumericType total = 0.;

    // theta_k appearing explicitly as a coverage factor; the free part is intact
    for (size_t a = 0; a < r.coverageFactors.size(); ++a) {
      const auto &f = r.coverageFactors[a];
      if (f.index != kIdx)
        continue;
      NumericType term = base * f.exponent * ipow(theta[kIdx], f.exponent - 1);
      for (size_t b = 0; b < r.coverageFactors.size(); ++b) {
        if (b == a)
          continue;
        const auto &g = r.coverageFactors[b];
        term *= ipow(theta[g.index], g.exponent);
      }
      term *= freePart(r, free);
      total += term;
    }

    // through theta_*t of kIdx's own type t; d(theta_*t)/d(theta_k) = -1
    const int tk = coverageSite[kIdx];
    const int n = r.freeSiteExponent[tk];
    if (n) {
      NumericType term = -base * n * ipow(free[tk], n - 1);
      for (const auto &g : r.coverageFactors)
        term *= ipow(theta[g.index], g.exponent);
      for (int t = 0; t < numSiteTypes; ++t)
        if (t != tk && r.freeSiteExponent[t])
          term *= ipow(free[t], r.freeSiteExponent[t]);
      total += term;
    }

    return total;
  }

  // Newton solve of sum_j nu_ij r_j = 0, clamped to the physical simplex.
  // theta is used as the initial guess and overwritten with the solution.
  // The Newton step is damped to keep every coverage in [0,1] and each site
  // type's sum below 1, so a mechanism starting from a bare surface under a
  // large flux climbs in halvings and can need a few hundred steps. The loop
  // exits the moment the step falls below `tolerance`, so a generous cap costs
  // nothing where convergence is quick, and is the difference between a
  // converged answer and a stopped one where it is not.
  void solveCoverages(const std::vector<NumericType> &gamma,
                      const std::vector<NumericType> &k,
                      std::vector<NumericType> &theta,
                      int maxIterations = 500,
                      NumericType tolerance = 1e-13) const {
    const size_t n = coverageNames.size();
    if (n == 0)
      return;

    std::vector<NumericType> F(n), dx(n);
    std::vector<std::vector<NumericType>> J(n, std::vector<NumericType>(n));

    for (int iter = 0; iter < maxIterations; ++iter) {
      const std::vector<NumericType> free = freeFractions(theta);

      std::fill(F.begin(), F.end(), NumericType(0.));
      for (auto &row : J)
        std::fill(row.begin(), row.end(), NumericType(0.));

      for (size_t j = 0; j < reactions.size(); ++j) {
        const auto &r = reactions[j];
        const NumericType rj = rate(r, k[j], gamma, theta, free);
        for (size_t i = 0; i < n; ++i)
          if (r.nu[i] != 0.)
            F[i] += r.nu[i] * rj;
        for (size_t c = 0; c < n; ++c) {
          const NumericType d = dRate(r, k[j], gamma, theta, free, int(c));
          if (d != 0.)
            for (size_t i = 0; i < n; ++i)
              if (r.nu[i] != 0.)
                J[i][c] += r.nu[i] * d;
        }
      }

      // A coverage that no reaction touches on this material is unconstrained:
      // its row of the Jacobian is empty and its residual is zero, which makes
      // the system singular. Nothing produces it, so pin it to zero. This
      // happens whenever a per-material block switches every step that feeds a
      // species off, which is exactly how a chemistry is restricted.
      for (size_t i = 0; i < n; ++i) {
        bool empty = std::abs(F[i]) < 1e-30;
        for (size_t c = 0; c < n && empty; ++c)
          empty = std::abs(J[i][c]) < 1e-30;
        if (empty) {
          J[i][i] = 1.;
          F[i] = theta[i]; // negated below, so the step is dx_i = -theta_i
        }
      }

      for (size_t i = 0; i < n; ++i)
        F[i] = -F[i];
      if (!denseSolve(J, F, dx))
        break;

      // Damp so the step keeps each theta_i >= 0 and each type's sum <= 1.
      // A coverage already AT zero must not scale the step: it has no room to
      // give, so scaling by it would be scaling by nothing and the solve would
      // stand still while the rest of the system is still far from balance.
      // The clamp below holds it at zero instead.
      NumericType scale = 1.;
      for (size_t i = 0; i < n; ++i)
        if (theta[i] > 0. && theta[i] + dx[i] < 0. && dx[i] < 0.)
          scale = std::min(scale, NumericType(0.5) * theta[i] / (-dx[i]));
      for (int t = 0; t < numSiteTypes; ++t) {
        NumericType cur = 0., step = 0.;
        for (size_t i = 0; i < n; ++i)
          if (coverageSite[i] == t) {
            cur += theta[i];
            step += dx[i];
          }
        if (cur + scale * step > 1. && step > 0.)
          scale = std::min(scale, std::max(NumericType(0.),
                                           NumericType(0.5) * (1. - cur) / step));
      }

      NumericType delta = 0.;
      for (size_t i = 0; i < n; ++i) {
        const NumericType next =
            std::min(NumericType(1.),
                     std::max(NumericType(0.), theta[i] + scale * dx[i]));
        delta = std::max(delta, std::abs(next - theta[i]));
        theta[i] = next;
      }
      if (delta < tolerance)
        break;
    }
  }

  // One monolayer, as a flux-second. Fluxes are in 1e15 /cm2/s, so a rate r
  // moves the coverage by r/(S0/1e15) per second. A mechanism that declares no
  // site density falls back to 1e15 /cm2, which is the density the rest of the
  // unit convention already assumes: r = k*theta is only in the same units as
  // a flux if one monolayer is 1e15 /cm2.
  NumericType coverageScale() const {
    return siteDensity > 0. ? NumericType(1e15) / siteDensity : NumericType(1.);
  }

  // Transient coverage step, for an atomic layer process.
  //
  // The residual that solveCoverages drives to zero IS the time derivative, up
  // to the site density:
  //
  //     S0 dtheta_i/dt = sum_j nu_ij r_j(theta, Gamma)
  //
  // An ALD half-cycle has no steady state to solve for. It saturates, and
  // "it saturates" is a statement about dtheta/dt, not about dtheta/dt = 0: a
  // dose under a constant flux would deposit without end. So a cyclic process
  // integrates this instead of solving the balance, and the same mechanism,
  // the same rate laws and the same nu serve both paths.
  //
  // EXPONENTIAL EULER, not plain Euler. An ALD mechanism carries fast
  // intermediates that sit near zero -- here the iodine-bearing fragment the
  // dose puts down and the plasma strips within microseconds -- so the system
  // is stiff. Plain Euler oscillates such a species around zero and the clamp
  // at zero then injects coverage once per step, which makes the answer drift
  // FURTHER as the step is refined rather than converging. Splitting each
  // species into production and linear loss,
  //
  //     dtheta_i/dt = P_i - L_i theta_i,   theta_i(t+h) = P/L + (theta_i - P/L) e^{-L h}
  //
  // is exact where the loss is first order, unconditionally positive, and
  // converges cleanly at first order. L_i is formed as r_j/theta_i, which is
  // finite for a mass-action loss because r_j carries theta_i as a factor.
  //
  // `maxChange` is the accuracy knob: the net change allowed in one sub-step,
  // with the sub-step count scaling as its inverse. On the SiN PE-ALD
  // mechanism the error in the dose is about 5% at 1e-3 and about 1% at 2e-4.
  // `dt` is honoured exactly.
  // `grown`, when given, accumulates the thickness the solid-forming steps
  // deposit over dt, integrated on the SAME sub-steps as the coverages. It has
  // to be: the rate follows a coverage that can fall by orders of magnitude
  // within one pulse -- nitridation consumes what the dose put down in a
  // fraction of a second -- so sampling it once per caller step and
  // multiplying by dt overestimates the integral badly, and by a factor that
  // depends on how finely the caller happens to have divided the pulse.
  void stepCoverages(const std::vector<NumericType> &gamma,
                     const std::vector<NumericType> &k,
                     std::vector<NumericType> &theta, NumericType dt,
                     NumericType maxChange = 1e-3,
                     int maxSubSteps = 1000000,
                     NumericType *grown = nullptr,
                     Material material = Material(
                         BuiltInMaterial::Undefined)) const {
    const size_t n = coverageNames.size();
    if (n == 0 || dt <= 0.)
      return;

    const NumericType scale = coverageScale();
    // the smallest coverage a loss rate is divided by: r_j carries theta_i as
    // a factor, so the quotient is finite, and this only guards the division
    constexpr NumericType floor = std::numeric_limits<NumericType>::min();
    std::vector<NumericType> P(n), L(n);

    NumericType elapsed = 0.;
    for (int sub = 0; sub < maxSubSteps && elapsed < dt; ++sub) {
      const std::vector<NumericType> free = freeFractions(theta);

      std::fill(P.begin(), P.end(), NumericType(0.));
      std::fill(L.begin(), L.end(), NumericType(0.));
      for (size_t j = 0; j < reactions.size(); ++j) {
        const auto &r = reactions[j];
        const NumericType rj = rate(r, k[j], gamma, theta, free) * scale;
        if (rj == 0.)
          continue;
        for (size_t i = 0; i < n; ++i) {
          if (r.nu[i] > 0.)
            P[i] += r.nu[i] * rj;
          else if (r.nu[i] < 0.)
            L[i] += -r.nu[i] * rj / std::max(theta[i], floor);
        }
      }

      // the sub-step follows the NET rate. Limiting on the loss rate alone
      // would refuse to take a step wherever a fast intermediate sits at its
      // quasi-steady value, which is exactly where the exponential form is
      // already exact and a long step is safe.
      NumericType fastest = 0.;
      for (size_t i = 0; i < n; ++i)
        fastest = std::max(fastest, std::abs(P[i] - L[i] * theta[i]));

      NumericType h = dt - elapsed;
      if (fastest > 0.)
        h = std::min(h, maxChange / fastest);

      if (grown)
        *grown += growthRate(gamma, k, theta, material) * h;

      for (size_t i = 0; i < n; ++i) {
        NumericType next;
        if (L[i] * h > 1e-8) {
          const NumericType steady = P[i] / L[i];
          next = steady + (theta[i] - steady) * std::exp(-L[i] * h);
        } else {
          next = theta[i] + h * (P[i] - L[i] * theta[i]);
        }
        theta[i] = std::min(NumericType(1.), std::max(NumericType(0.), next));
      }
      clampToSimplex(theta);
      elapsed += h;
    }
  }

  // Hold each site type's occupancy at or below one. Euler can overshoot the
  // simplex where the steady-state solve was damped away from it, so the
  // coverages of an over-filled type are scaled back in proportion rather than
  // clipped one by one, which would silently favour whichever came first.
  void clampToSimplex(std::vector<NumericType> &theta) const {
    const size_t n = theta.size();
    for (int t = 0; t < numSiteTypes; ++t) {
      NumericType sum = 0.;
      for (size_t i = 0; i < n; ++i)
        if (coverageSite[i] == t)
          sum += theta[i];
      if (sum > 1.) {
        for (size_t i = 0; i < n; ++i)
          if (coverageSite[i] == t)
            theta[i] /= sum;
      }
    }
  }

  // The surface velocity: every solid-forming step contributes its atom count
  // divided by the density of ITS OWN solid, so a mechanism depositing a
  // polymer while etching a substrate adds the two motions correctly.
  NumericType growthRate(const std::vector<NumericType> &gamma,
                         const std::vector<NumericType> &k,
                         const std::vector<NumericType> &theta,
                         Material material) const {
    const std::vector<NumericType> free = freeFractions(theta);
    NumericType total = 0.;
    for (size_t j = 0; j < reactions.size(); ++j)
      if (reactions[j].solidAtoms != 0.)
        total += reactions[j].solidAtoms *
                 rate(reactions[j], k[j], gamma, theta, free) /
                 densityOf(reactions[j].solidIndex, material);
    return total;
  }

  NumericType growthRate(const std::vector<NumericType> &gamma,
                         const std::vector<NumericType> &k,
                         const std::vector<NumericType> &theta) const {
    return growthRate(gamma, k, theta, Material(BuiltInMaterial::Undefined));
  }

  // The incident flux of every gas species, as the reaction file supplies it.
  // An ion yield channel carries no flux of its own: the particle produces one
  // by weighting the ion flux with the yield, so what stands here is that yield
  // at NORMAL incidence, which is what a blanket surface sees. Inside a feature
  // the ray tracer resolves it per ray, and this is only the analytic estimate.
  std::vector<NumericType>
  sourceFluxes(Material material = Material(BuiltInMaterial::Undefined)) const {
    std::vector<NumericType> gamma;
    gamma.reserve(gas.size());
    for (const auto &g : gas)
      gamma.push_back(g.sourceFlux);
    for (const auto &y : ionYields)
      gamma[y.gasIndex] *=
          y.materialA.get(material) *
          std::max(std::sqrt(ionSource.meanEnergy) -
                       std::sqrt(y.materialEth.get(material)),
                   NumericType(0.));
    return gamma;
  }

  // The density of solid `s` under a point of `material`.
  NumericType densityOf(int s, Material material) const {
    if (s < 0 || s >= static_cast<int>(solids.size()))
      return rhoSolid; // no solid was declared; the single-density form
    const auto &sp = solids[s];
    return sp.materialDependent ? sp.rhoMaterial.get(material) : sp.rho;
  }

  int addSolid(const std::string &name, NumericType rho) {
    SolidPhase sp;
    sp.name = name;
    sp.rho = rho;
    solids.push_back(sp);
    if (solids.size() == 1)
      rhoSolid = rho;
    return static_cast<int>(solids.size()) - 1;
  }

  void setSolidDensity(int solidIndex, Material material, NumericType rho) {
    auto &sp = solids.at(solidIndex);
    if (!sp.materialDependent) {
      sp.rhoMaterial = MaterialValueMap<NumericType>::fromDefault(sp.rho);
      sp.materialDependent = true;
    }
    sp.rhoMaterial.set(material, rho);
  }

  // The reaction as written in the file, carried for reporting only.
  void setEquation(int reactionIndex, const std::string &equation) {
    reactions.at(reactionIndex).equation = equation;
  }

  void setSolidAtoms(int reactionIndex, int solidIndex, NumericType atoms) {
    auto &r = reactions.at(reactionIndex);
    r.solidAtoms = atoms;
    r.solidIndex = solidIndex;
  }

  auto toProcessMetaData() const {
    std::unordered_map<std::string, std::vector<double>> md;
    md["Temperature"] = {static_cast<double>(temperature)};
    md["RhoSolid"] = {static_cast<double>(rhoSolid)};
    md["NumCoverages"] = {static_cast<double>(coverageNames.size())};
    md["NumReactions"] = {static_cast<double>(reactions.size())};
    auto k = rateConstants();
    for (size_t j = 0; j < k.size(); ++j)
      md["k_" + std::to_string(j + 1)] = {static_cast<double>(k[j])};
    for (size_t g = 0; g < gas.size(); ++g)
      if (gas[g].traced) {
        md["Flux_" + gas[g].label] = {static_cast<double>(gas[g].sourceFlux)};
        md["Sticking_" + gas[g].label] = {
            static_cast<double>(stickingOf(int(g)))};
      }
    return md;
  }

private:
  // Gaussian elimination with partial pivoting; the system is tiny.
  static bool denseSolve(std::vector<std::vector<NumericType>> A,
                         std::vector<NumericType> b,
                         std::vector<NumericType> &x) {
    const size_t n = b.size();
    x.assign(n, NumericType(0.));
    for (size_t c = 0; c < n; ++c) {
      size_t p = c;
      for (size_t r = c + 1; r < n; ++r)
        if (std::abs(A[r][c]) > std::abs(A[p][c]))
          p = r;
      if (std::abs(A[p][c]) < 1e-300)
        return false;
      std::swap(A[c], A[p]);
      std::swap(b[c], b[p]);
      for (size_t r = c + 1; r < n; ++r) {
        const NumericType f = A[r][c] / A[c][c];
        if (f == 0.)
          continue;
        for (size_t cc = c; cc < n; ++cc)
          A[r][cc] -= f * A[c][cc];
        b[r] -= f * b[c];
      }
    }
    for (size_t i = n; i-- > 0;) {
      NumericType s = b[i];
      for (size_t c = i + 1; c < n; ++c)
        s -= A[i][c] * x[c];
      x[i] = s / A[i][i];
    }
    return true;
  }
};

namespace impl {

template <typename NumericType, int D>
class ChemicalSurfaceModel : public SurfaceModel<NumericType> {
public:
  using SurfaceModel<NumericType>::coverages;
  using SurfaceModel<NumericType>::surfaceData;
  const ChemicalMechanism<NumericType> &mech;

  explicit ChemicalSurfaceModel(const ChemicalMechanism<NumericType> &m)
      : mech(m) {}

  // Zero means solve the steady state; positive means integrate for this long.
  // The strategy sets it once per sub-step of a pulse.
  void setTimeStep(NumericType dt) override { timeStep_ = dt; }

  // An atomic layer process carries the coverages of one step into the next:
  // what the dose leaves on the surface is what the plasma acts on. The
  // strategy re-initialises coverages once per cycle, so without this the
  // surface would be wiped between steps and every step would start bare.
  void setPreserveCoverages(bool preserve) {
    preserveCoverages_ = preserve;
    atomicLayer_ = preserve;
  }

  // The transient integrator's accuracy: the largest coverage change one
  // internal sub-step may take. First order, so halving it roughly halves
  // the integration error; the sub-step count scales with its inverse. On
  // the SiN PE-ALD mechanism 1e-3 integrates a dose to about 5%, 2e-4 to
  // about 1%.
  void setMaxCoverageChange(NumericType maxChange) {
    maxCoverageChange_ = maxChange;
  }

  // An atomic layer process re-initialises coverages once per cycle, after
  // the advection has moved the surface and the disk mesh has been rebuilt.
  // The termination the previous cycle left is physical state, so it is
  // carried onto the new mesh by interpolation: a KD-tree over the previous
  // points, and an inverse-distance average of the k nearest for every new
  // point. Point INDICES carry no meaning across a re-mesh -- an insertion
  // anywhere shifts every index after it, and an equal point count is
  // coincidence -- so the transfer is geometric, and runs whenever previous
  // coverages exist. A convex combination of valid coverages stays on the
  // physical simplex, so the result needs no clamping.
  void initializeCoverages(
      unsigned numGeometryPoints,
      const std::vector<Vec3D<NumericType>> &coordinates) override {
    if (mech.coverageNames.empty())
      return;
    if (preserveCoverages_ && coverages != nullptr &&
        !coveragePoints_.empty() && holdsAll(coveragePoints_.size())) {
      transferCoverages(coordinates);
    } else {
      initializeZero(numGeometryPoints);
    }
    coveragePoints_ = coordinates;
    grown_.assign(numGeometryPoints, 0.);
  }

  void initializeCoverages(unsigned numGeometryPoints) override {
    // A mechanism without surface intermediates (pure sputtering, say) has no
    // coverage loop to run. Leaving the container null tells the strategy so;
    // creating an empty one would make it iterate over nothing, and the GPU
    // engine would then try to allocate a zero-byte coverage buffer.
    if (mech.coverageNames.empty())
      return;
    initializeZero(numGeometryPoints);
    coveragePoints_.clear();
    grown_.assign(numGeometryPoints, 0.);
  }

  void initializeZero(unsigned numGeometryPoints) {
    if (coverages == nullptr)
      coverages = PointData<NumericType>::New();
    else
      coverages->clear();
    std::vector<NumericType> zero(numGeometryPoints, 0.);
    for (const auto &name : mech.coverageNames)
      coverages->insertNextScalarData(zero, name);
  }

  // Interpolate every coverage from the previous mesh onto the new one.
  void transferCoverages(const std::vector<Vec3D<NumericType>> &newPoints) {
    const size_t nCov = mech.coverageNames.size();
    const size_t nNew = newPoints.size();
    constexpr int kNeighbors = 3;

    KDTree<NumericType, Vec3D<NumericType>> tree;
    tree.setPoints(coveragePoints_);
    tree.build();

    std::vector<const std::vector<NumericType> *> oldCov(nCov);
    std::vector<std::vector<NumericType>> newCov(
        nCov, std::vector<NumericType>(nNew, 0.));
    for (size_t i = 0; i < nCov; ++i)
      oldCov[i] = coverages->getScalarData(mech.coverageNames[i]);

#pragma omp parallel for
    for (size_t p = 0; p < nNew; ++p) {
      const auto found = tree.findKNearest(newPoints[p], kNeighbors);
      if (!found || found->empty())
        continue;
      NumericType weightSum = 0.;
      for (const auto &[idx, dist] : *found) {
        // an exact hit dominates; the epsilon keeps the weight finite there
        const NumericType w = NumericType(1.) / (dist + NumericType(1e-12));
        weightSum += w;
        for (size_t i = 0; i < nCov; ++i)
          newCov[i][p] += w * oldCov[i]->at(idx);
      }
      for (size_t i = 0; i < nCov; ++i)
        newCov[i][p] /= weightSum;
    }

    coverages->clear();
    for (size_t i = 0; i < nCov; ++i)
      coverages->insertNextScalarData(std::move(newCov[i]),
                                      mech.coverageNames[i]);
  }

  // True when the coverage container already carries every coverage of this
  // mechanism, at this point count.
  bool holdsAll(unsigned numGeometryPoints) const {
    for (const auto &name : mech.coverageNames) {
      const auto *data = coverages->getScalarData(name);
      if (!data || data->size() != numGeometryPoints)
        return false;
    }
    return true;
  }

  void initializeSurfaceData(unsigned numGeometryPoints) override {
    if (!Logger::hasIntermediate())
      return;
    if (surfaceData == nullptr)
      surfaceData = PointData<NumericType>::New();
    else
      surfaceData->clear();
    std::vector<NumericType> zero(numGeometryPoints, 0.);
    surfaceData->insertNextScalarData(zero, "growthRate");
  }

  void updateCoverages(SmartPointer<PointData<NumericType>> fluxes,
                       const std::vector<NumericType> &materialIds) override {
    const auto numPoints = materialIds.size();
    const auto k = mech.rateConstants();
    // one set of constants per material present, computed once
    const auto kByMaterial = ratesByMaterial(mech, materialIds);
    const size_t nCov = mech.coverageNames.size();

    // pointers to the recorded flux of each traced gas species
    std::vector<std::vector<NumericType> *> fluxPtr(mech.gas.size(), nullptr);
    for (size_t g = 0; g < mech.gas.size(); ++g)
      if (mech.gas[g].traced)
        fluxPtr[g] = fluxes->getScalarData(mech.gas[g].label);

    std::vector<std::vector<NumericType> *> cov(nCov, nullptr);
    for (size_t i = 0; i < nCov; ++i) {
      cov[i] = coverages->getScalarData(mech.coverageNames[i]);
      cov[i]->resize(numPoints);
    }

#pragma omp parallel
    {
      std::vector<NumericType> gamma(mech.gas.size(), 0.);
      std::vector<NumericType> theta(nCov, 0.);
#pragma omp for
      for (size_t p = 0; p < numPoints; ++p) {
        for (size_t g = 0; g < mech.gas.size(); ++g)
          gamma[g] = fluxPtr[g] ? fluxPtr[g]->at(p) * mech.gas[g].sourceFlux : 0.;
        // warm start from the previous solution at this point
        for (size_t i = 0; i < nCov; ++i)
          theta[i] = cov[i]->at(p);
        // a chemistry that differs between materials uses the constants of the
        // material under this point
        const auto &kHere = ratesAt(kByMaterial, k, materialIds[p]);
        if (timeStep_ > 0.) {
          // An atomic layer process advects ONCE per cycle, by what the cycle
          // deposited. That is the integral of the rate over the steps, not
          // the rate at the end of them -- by the end of a cycle the last
          // purge is running and nothing is flowing, so the instantaneous
          // rate is zero however much the cycle grew.
          NumericType deposited = 0.;
          mech.stepCoverages(gamma, kHere, theta, timeStep_,
                             maxCoverageChange_, 1000000, &deposited,
                             MaterialMap::mapToMaterial(materialIds[p]));
          accumulate(p, deposited * unitConversion());
        } else {
          mech.solveCoverages(gamma, kHere, theta);
        }
        for (size_t i = 0; i < nCov; ++i)
          cov[i]->at(p) = theta[i];
      }
    }
  }

  // A purge is this same mechanism with nothing flowing: the thermal steps
  // keep running -- HI leaving, H2 recombining off the surface -- while every
  // source flux is zero. The desorption fluxes the strategy hands in describe
  // re-adsorption of what just left, which this model does not yet resolve;
  // ignoring them makes the purge purely thermal, which is the conservative
  // reading and the one the reaction file already expresses.
  void updateCoveragesFromDesorption(
      SmartPointer<PointData<NumericType>>,
      const std::vector<NumericType> &materialIds) override {
    if (timeStep_ <= 0. || mech.coverageNames.empty() || coverages == nullptr)
      return;

    const auto numPoints = materialIds.size();
    const auto k = mech.rateConstants();
    const auto kByMaterial = ratesByMaterial(mech, materialIds);
    const size_t nCov = mech.coverageNames.size();

    std::vector<std::vector<NumericType> *> cov(nCov, nullptr);
    for (size_t i = 0; i < nCov; ++i) {
      cov[i] = coverages->getScalarData(mech.coverageNames[i]);
      cov[i]->resize(numPoints);
    }

    const std::vector<NumericType> noFlux(mech.gas.size(), 0.);

#pragma omp parallel
    {
      std::vector<NumericType> theta(nCov, 0.);
#pragma omp for
      for (size_t p = 0; p < numPoints; ++p) {
        for (size_t i = 0; i < nCov; ++i)
          theta[i] = cov[i]->at(p);
        const auto &kHere = ratesAt(kByMaterial, k, materialIds[p]);
        NumericType deposited = 0.;
        mech.stepCoverages(noFlux, kHere, theta, timeStep_,
                           maxCoverageChange_, 1000000, &deposited,
                           MaterialMap::mapToMaterial(materialIds[p]));
        accumulate(p, deposited * unitConversion());
        for (size_t i = 0; i < nCov; ++i)
          cov[i]->at(p) = theta[i];
      }
    }
  }

  // Rate constants for every material present on the surface, computed once.
  static std::unordered_map<int, std::vector<NumericType>>
  ratesByMaterial(const ChemicalMechanism<NumericType> &mech,
                  const std::vector<NumericType> &materialIds) {
    std::unordered_map<int, std::vector<NumericType>> byMaterial;
    if (!mech.materialDependent)
      return byMaterial;
    for (auto id : materialIds) {
      const int key = static_cast<int>(id);
      if (byMaterial.find(key) == byMaterial.end())
        byMaterial.emplace(key,
                           mech.rateConstantsFor(MaterialMap::mapToMaterial(id)));
    }
    return byMaterial;
  }

  static const std::vector<NumericType> &
  ratesAt(const std::unordered_map<int, std::vector<NumericType>> &byMaterial,
          const std::vector<NumericType> &fallback, NumericType materialId) {
    if (byMaterial.empty())
      return fallback;
    return byMaterial.at(static_cast<int>(materialId));
  }

  // nm/s as the mechanism reports it, in the process's own units.
  static double unitConversion() {
    return units::Time::convertSecond() / units::Length::convertNanometer();
  }

  // Thickness deposited at one point since the last time this was read.
  void accumulate(size_t point, NumericType amount) {
    if (grown_.size() > point)
      grown_[point] += amount;
  }

  SmartPointer<std::vector<NumericType>>
  calculateVelocities(SmartPointer<PointData<NumericType>> fluxes,
                      const std::vector<Vec3D<NumericType>> &coordinates,
                      const std::vector<NumericType> &materialIds) override {
    const auto numPoints = materialIds.size();

    // In a cyclic process this is asked once per cycle, immediately before the
    // advection, and the strategy advects for one time unit -- so what it
    // wants is the thickness this cycle deposited, not a rate. Hand it over
    // and start the next cycle from zero.
    if (atomicLayer_) {
      grown_.resize(numPoints, 0.);
      auto perCycle = SmartPointer<std::vector<NumericType>>::New(grown_);
      // The fastest-growing point is the open field, where nothing shadows
      // the flux, so this running total is the blanket growth per cycle --
      // the quantity a saturation curve is measured as, and the one to check
      // a mechanism against before trusting what it does inside a feature.
      if (!grown_.empty())
        fieldGrowth_ += *std::max_element(grown_.begin(), grown_.end());
      ++cyclesRun_;
      std::fill(grown_.begin(), grown_.end(), NumericType(0.));
      return perCycle;
    }
    std::vector<NumericType> velocity(numPoints, 0.);
    const auto k = mech.rateConstants();
    // one set of constants per material present, computed once
    const auto kByMaterial = ratesByMaterial(mech, materialIds);
    const size_t nCov = mech.coverageNames.size();

    std::vector<std::vector<NumericType> *> fluxPtr(mech.gas.size(), nullptr);
    for (size_t g = 0; g < mech.gas.size(); ++g)
      if (mech.gas[g].traced)
        fluxPtr[g] = fluxes->getScalarData(mech.gas[g].label);

    std::vector<const std::vector<NumericType> *> cov(nCov, nullptr);
    for (size_t i = 0; i < nCov; ++i)
      cov[i] = coverages->getScalarData(mech.coverageNames[i]);

    std::vector<NumericType> *growth = nullptr;
    if (Logger::hasIntermediate()) {
      growth = surfaceData->getScalarData("growthRate");
      growth->resize(numPoints);
    }

    const double unitConversion =
        units::Time::convertSecond() / units::Length::convertNanometer();

#pragma omp parallel
    {
      std::vector<NumericType> gamma(mech.gas.size(), 0.);
      std::vector<NumericType> theta(nCov, 0.);
#pragma omp for
      for (size_t p = 0; p < numPoints; ++p) {
        for (size_t g = 0; g < mech.gas.size(); ++g)
          gamma[g] = fluxPtr[g] ? fluxPtr[g]->at(p) * mech.gas[g].sourceFlux : 0.;
        for (size_t i = 0; i < nCov; ++i)
          theta[i] = cov[i]->at(p);
        // the growth rate follows the material under each point, both in the
        // rate constants and in the density the removal is divided by
        velocity[p] =
            mech.growthRate(gamma, ratesAt(kByMaterial, k, materialIds[p]),
                            theta,
                            MaterialMap::mapToMaterial(materialIds[p])) *
            unitConversion;
        if (growth)
          growth->at(p) = velocity[p];
      }
    }

    return SmartPointer<std::vector<NumericType>>::New(std::move(velocity));
  }

  std::optional<std::unordered_map<std::string, NumericType>>
  getDiffusionCoefficients() const override {
    std::unordered_map<std::string, NumericType> d;
    for (size_t i = 0; i < mech.coverageNames.size(); ++i)
      if (mech.diffusionCoefficients[i] > 0.)
        d[mech.coverageNames[i]] = mech.diffusionCoefficients[i];
    if (d.empty())
      return std::nullopt;
    return d;
  }

private:
  NumericType timeStep_ = 0.;       // 0 = steady state, > 0 = integrate
  NumericType maxCoverageChange_ = 1e-3; // sub-step accuracy of stepCoverages
  bool preserveCoverages_ = false;  // carry coverages across process steps
  bool atomicLayer_ = false;        // advect once per cycle, by what it grew
  std::vector<NumericType> grown_;  // thickness deposited this cycle, per point
  NumericType fieldGrowth_ = 0.;    // total on the open field, over all cycles
  unsigned cyclesRun_ = 0;
  std::vector<Vec3D<NumericType>> coveragePoints_; // mesh the coverages live on

public:
  // Blanket film grown so far, and over how many cycles: their ratio is the
  // growth per cycle a flat-surface measurement would report.
  NumericType getFieldGrowth() const { return fieldGrowth_; }
  unsigned getCyclesRun() const { return cyclesRun_; }

private:
};

// Neutral gas species. Records the raw incident flux; its re-emission uses the
// coverage-dependent sticking of the adsorption step that consumes it.
template <typename NumericType, int D>
class ChemicalParticle final
    : public viennaray::Particle<ChemicalParticle<NumericType, D>, NumericType> {
  const std::string fluxLabel;
  // sticking already evaluated at the mechanism temperature, per material
  const MaterialValueMap<NumericType> sticking;
  const int freeSiteExponent;
  const std::vector<int> siteCoverages; // coverage indices on this site type
  const NumericType sourcePower;

public:
  ChemicalParticle(std::string label, MaterialValueMap<NumericType> stickingAtT,
                   int freeSiteExp, std::vector<int> siteCoverages,
                   NumericType power = 1.)
      : fluxLabel(std::move(label)), sticking(std::move(stickingAtT)),
        freeSiteExponent(freeSiteExp), siteCoverages(std::move(siteCoverages)),
        sourcePower(power) {}

  void surfaceCollision(NumericType rayWeight, const Vec3D<NumericType> &,
                        const Vec3D<NumericType> &, const unsigned int primID,
                        const int, PointData<NumericType> &localData,
                        const PointData<NumericType> *, RNG &) override {
    localData.addToScalarData(0, primID, rayWeight);
  }

  std::pair<NumericType, Vec3D<NumericType>>
  surfaceReflection(NumericType, const Vec3D<NumericType> &,
                    const Vec3D<NumericType> &geomNormal,
                    const unsigned int primID, const int materialId,
                    const PointData<NumericType> *globalData,
                    RNG &rngState) override {
    NumericType occupied = 0.;
    for (int i : siteCoverages)
      occupied += globalData->getScalarData(i)->at(primID);
    NumericType free = std::max(NumericType(1.) - occupied, NumericType(0.));

    // the sticking follows the material under the hit, so a species that
    // adsorbs on one material and not another reflects accordingly
    NumericType sEff = sticking.get(MaterialMap::mapToMaterial(materialId));
    for (int i = 0; i < freeSiteExponent; ++i)
      sEff *= free;

    auto direction =
        viennaray::ReflectionDiffuse<NumericType, D>(geomNormal, rngState);
    return std::pair<NumericType, Vec3D<NumericType>>{sEff, direction};
  }

  NumericType getSourceDistributionPower() const override {
    return sourcePower;
  }
  [[nodiscard]] std::vector<std::string> getLocalDataLabels() const override {
    return {fluxLabel};
  }
};

// An ion. It carries an energy, loses some of it on each glancing reflection,
// and at every hit deposits a YIELD-WEIGHTED flux into one channel per
// ion-driven reaction:
//
//   Y = A(material) * max(sqrt(E) - sqrt(Eth(material)), 0) * f(theta)
//
// Because the yield is folded into the flux here, the surface chemistry treats
// an ion channel exactly like a neutral flux, and the rate law stays mass
// action. Nothing in the solver knows about ions.
template <typename NumericType, int D>
class ChemicalIon final
    : public viennaray::Particle<ChemicalIon<NumericType, D>, NumericType> {
  using IonSource = typename ChemicalMechanism<NumericType>::IonSource;
  using IonYield = typename ChemicalMechanism<NumericType>::IonYield;

  const IonSource source;
  const std::vector<IonYield> yields;
  const NumericType A_energy;
  const NumericType minEth; // stop the ion once it can drive nothing
  NumericType E = 0.;

public:
  ChemicalIon(IonSource src, std::vector<IonYield> ys)
      : source(src), yields(std::move(ys)),
        A_energy(1. / (1. + src.n_l * (M_PI_2 / (src.inflectAngle * M_PI /
                                                 180.) -
                                       1.))),
        minEth([&] {
          NumericType m = std::numeric_limits<NumericType>::max();
          for (const auto &y : yields)
            m = std::min(m, y.Eth);
          return yields.empty() ? NumericType(0.) : m;
        }()) {}

  void surfaceCollision(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                        const Vec3D<NumericType> &geomNormal,
                        const unsigned int primID, const int materialId,
                        PointData<NumericType> &localData,
                        const PointData<NumericType> *, RNG &) override {
    const auto material = MaterialMap::mapToMaterial(materialId);
    const NumericType cosTheta =
        util::saturate(-DotProduct(rayDir, geomNormal));
    const NumericType angle = std::acos(cosTheta);
    const NumericType sqrtE = std::sqrt(E);

    // the two angular forms of the yield, as used for sputtering and for
    // ion-enhanced etching
    for (size_t c = 0; c < yields.size(); ++c) {
      const auto &y = yields[c];
      const NumericType A = y.materialA.get(material);
      const NumericType Eth = y.materialEth.get(material);

      NumericType f;
      if (y.enhanced) {
        f = cosTheta < 0.5 ? std::max(NumericType(3.) - NumericType(6.) * angle /
                                                            NumericType(M_PI),
                                      NumericType(0.))
                           : NumericType(1.);
      } else {
        f = std::max((NumericType(1.) +
                      y.B * (NumericType(1.) - cosTheta * cosTheta)) *
                         cosTheta,
                     NumericType(0.));
      }

      const NumericType Y =
          A * std::max(sqrtE - std::sqrt(Eth), NumericType(0.)) * f;
      localData.addToScalarData(int(c), primID, Y * rayWeight);
    }
  }

  std::pair<NumericType, Vec3D<NumericType>>
  surfaceReflection(NumericType, const Vec3D<NumericType> &rayDir,
                    const Vec3D<NumericType> &geomNormal, const unsigned int,
                    const int, const PointData<NumericType> *,
                    RNG &rngState) override {
    const NumericType cosTheta =
        util::saturate(-DotProduct(rayDir, geomNormal));
    const NumericType incAngle = std::acos(cosTheta);

    // a steep hit is absorbed; a glancing one reflects
    const NumericType thetaRMin = source.thetaRMin * M_PI / 180.;
    const NumericType thetaRMax = source.thetaRMax * M_PI / 180.;
    NumericType sticking = 1.;
    if (incAngle > thetaRMin)
      sticking = 1. - util::saturate((incAngle - thetaRMin) /
                                     (thetaRMax - thetaRMin));
    if (sticking >= 1.)
      return VIENNARAY_PARTICLE_STOP;

    const NumericType newEnergy =
        updateEnergy(rngState, E, incAngle, A_energy,
                     source.inflectAngle * M_PI / 180., source.n_l);
    if (newEnergy <= minEth)
      return VIENNARAY_PARTICLE_STOP;

    E = newEnergy;
    auto direction = viennaray::ReflectionConedCosine<NumericType, D>(
        rayDir, geomNormal, rngState,
        M_PI_2 - std::min(incAngle, NumericType(source.minAngle * M_PI / 180.)));
    return std::pair<NumericType, Vec3D<NumericType>>{sticking, direction};
  }

  void initNew(RNG &rngState) override {
    E = initNormalDistEnergy(rngState, source.meanEnergy, source.sigmaEnergy);
  }

  NumericType getSourceDistributionPower() const override {
    return source.exponent;
  }

  [[nodiscard]] std::vector<std::string> getLocalDataLabels() const override {
    std::vector<std::string> labels;
    labels.reserve(yields.size());
    for (const auto &y : yields)
      labels.push_back(y.label);
    return labels;
  }
};

} // namespace impl

// Device-side data for the particles. Mirrors the definition in
// gpu/models/SurfaceChemistry.cuh; the layouts must agree.
struct SurfaceChemistryParamsGPU {
  static constexpr int maxParticles = 16; // a published mechanism
                                          // can adsorb a dozen species
  static constexpr int maxCoverages = 16;
  static constexpr int maxMaterials = 8; // per-particle sticking overrides

  int numCoverages = 0;
  int coverageSite[maxCoverages] = {}; // site-type index of each coverage
  int freeSiteExponent[maxParticles] = {};
  int stickingSite[maxParticles] = {}; // site type each particle sticks to

  // Sticking per particle, already evaluated at the mechanism temperature.
  // `defaultSticking` applies unless the material under the hit is listed.
  float defaultSticking[maxParticles] = {};
  int numOverrides[maxParticles] = {};
  int overrideMaterial[maxParticles][maxMaterials] = {}; // legacy material ids
  float overrideSticking[maxParticles][maxMaterials] = {};

  // --- ions -----------------------------------------------------------------
  // Everything below comes from the reaction file, by way of the mechanism.
  static constexpr int maxYields = 6;

  float meanEnergy = 100.f, sigmaEnergy = 10.f;
  float inflectAngle = 89.f, n_l = 10.f; // radians once uploaded
  float minAngle = 80.f, thetaRMin = 70.f, thetaRMax = 90.f;
  float minEth = 0.f; // stop the ion once it can drive nothing

  int numYields = 0;
  float yieldA[maxYields] = {};
  float yieldEth[maxYields] = {};
  float yieldB[maxYields] = {};
  int yieldEnhanced[maxYields] = {};
  // per-material A and Eth, so a mask is harder to sputter
  int yieldNumOverrides[maxYields] = {};
  int yieldOverrideMaterial[maxYields][maxMaterials] = {};
  float yieldOverrideA[maxYields][maxMaterials] = {};
  float yieldOverrideEth[maxYields][maxMaterials] = {};
};

// The host and the device each carry their own copy of this struct, and a
// difference between them is silent: the device reads the right bytes at the
// wrong offsets, so a coverage's site index becomes garbage and the chemistry
// quietly changes. Both copies assert the same shape, so editing one without
// the other fails the build instead.
static_assert(SurfaceChemistryParamsGPU::maxParticles == 16 &&
                  SurfaceChemistryParamsGPU::maxCoverages == 16 &&
                  SurfaceChemistryParamsGPU::maxMaterials == 8 &&
                  SurfaceChemistryParamsGPU::maxYields == 6,
              "SurfaceChemistryParamsGPU must have the same shape in "
              "psSurfaceChemistry.hpp and gpu/models/SurfaceChemistry.cuh");

#ifdef VIENNACORE_COMPILE_GPU
namespace gpu {
/// The same mechanism, traced on the GPU. The coverage solve stays on the CPU
/// in ChemicalSurfaceModel; only the transport moves to the device, so the
/// chemistry is shared bit for bit with the CPU model.
template <typename NumericType, int D>
class SurfaceChemistry : public ProcessModelGPU<NumericType, D> {
  ::viennaps::ChemicalMechanism<NumericType> mech_; // the current phase
  std::vector<std::pair<std::string, ::viennaps::ChemicalMechanism<NumericType>>>
      phaseMechanisms_;
  NumericType maxCoverageChange_ = 1e-3;
  SurfaceChemistryParamsGPU deviceParams_;
  SmartPointer<impl::ChemicalSurfaceModel<NumericType, D>> surfModel_ = nullptr;

public:
  SurfaceChemistry() = default;

  explicit SurfaceChemistry(const ::viennaps::ChemicalMechanism<NumericType> &m)
      : mech_(m) {
    initializeModel();
  }

  // The chemistry of one step of an atomic layer cycle. As on the CPU, the
  // steps of one cycle must describe the same surface, since the coverages of
  // one are the initial condition of the next.
  void addMechanism(const std::string &name,
                    const ::viennaps::ChemicalMechanism<NumericType> &m) {
    if (!phaseMechanisms_.empty() &&
        m.coverageNames != phaseMechanisms_.front().second.coverageNames) {
      VIENNACORE_LOG_ERROR(
          "Mechanism '" + name + "' declares different coverages than '" +
          phaseMechanisms_.front().first +
          "'. The steps of one cycle must describe the same surface.");
      return;
    }
    phaseMechanisms_.emplace_back(name, m);
    if (phaseMechanisms_.size() == 1)
      mech_ = m;
    initializeModel();
  }

  // Select the chemistry and the flowing species for one step. Only the host
  // side moves: the coverage solve reads mech_, and the device already holds
  // every particle of every step with its own sticking, so nothing is
  // re-uploaded and no shader is rebuilt between steps of a cycle.
  void setActivePhase(const std::string &phaseName,
                      const std::vector<std::string> &activeSpecies,
                      const std::string &mechanismName) override {
    if (!mechanismName.empty()) {
      const auto it = std::find_if(
          phaseMechanisms_.begin(), phaseMechanisms_.end(),
          [&](const auto &entry) { return entry.first == mechanismName; });
      if (it == phaseMechanisms_.end()) {
        VIENNACORE_LOG_ERROR("Phase '" + phaseName + "' names mechanism '" +
                             mechanismName + "', which was not registered.");
        return;
      }
      mech_ = it->second;
    }
    const bool all = activeSpecies.size() == 1 && activeSpecies.front() == "*";
    for (auto &g : mech_.gas) {
      if (g.label.empty())
        continue;
      const bool flowing =
          all || std::find(activeSpecies.begin(), activeSpecies.end(),
                           g.label) != activeSpecies.end();
      g.sourceFlux = flowing ? sourceFluxOf(g.label) : NumericType(0.);
    }
  }

  void setAtomicLayerProcess(bool enable = true) {
    this->isALP = enable;
    if (surfModel_)
      surfModel_->setPreserveCoverages(enable);
  }

  // Accuracy of the transient coverage integration; see
  // ChemicalSurfaceModel::setMaxCoverageChange.
  void setMaxCoverageChange(NumericType maxChange) {
    if (surfModel_)
      surfModel_->setMaxCoverageChange(maxChange);
    maxCoverageChange_ = maxChange;
  }

  ::viennaps::ChemicalMechanism<NumericType> &getMechanism() { return mech_; }

  NumericType growthPerCycle() const {
    if (!surfModel_ || surfModel_->getCyclesRun() == 0)
      return 0.;
    return surfModel_->getFieldGrowth() / surfModel_->getCyclesRun();
  }

private:
  NumericType sourceFluxOf(const std::string &label) const {
    for (const auto &entry : phaseMechanisms_)
      for (const auto &g : entry.second.gas)
        if (g.label == label)
          return g.sourceFlux;
    return NumericType(0.);
  }

  std::vector<const ::viennaps::ChemicalMechanism<NumericType> *>
  mechanismsToTrace() const {
    std::vector<const ::viennaps::ChemicalMechanism<NumericType> *> out;
    if (phaseMechanisms_.empty()) {
      out.push_back(&mech_);
    } else {
      for (const auto &entry : phaseMechanisms_)
        out.push_back(&entry.second);
    }
    return out;
  }

  void initializeModel() {
    // addMechanism rebuilds the device setup, so the particles of the previous
    // build must go; the base keeps the vector private but hands out a
    // reference to it.
    this->getParticleTypes().clear();
    deviceParams_ = SurfaceChemistryParamsGPU{};
    const int nCov = static_cast<int>(mech_.coverageNames.size());
    deviceParams_.numCoverages = nCov;
    if (nCov > SurfaceChemistryParamsGPU::maxCoverages)
      VIENNACORE_LOG_ERROR("SurfaceChemistry GPU: too many coverages.");
    for (int i = 0; i < nCov && i < SurfaceChemistryParamsGPU::maxCoverages;
         ++i)
      deviceParams_.coverageSite[i] = mech_.coverageSite[i];

    std::unordered_map<std::string, unsigned> pMap;
    std::vector<viennaray::gpu::CallableConfig> cMap;

    // One particle per traced species over EVERY registered mechanism. The
    // shader is built once and cannot change as the cycle moves from step to
    // step, so the device holds all of them; a species that is not flowing in
    // the current step is silenced by its source flux on the host instead.
    unsigned p = 0;
    std::vector<std::string> seen;
    for (const auto *source : mechanismsToTrace())
    for (size_t g = 0; g < source->gas.size(); ++g) {
      if (!source->gas[g].traced || source->gas[g].isIonChannel)
        continue; // an ion channel is written by the ion, not by a neutral
      if (std::find(seen.begin(), seen.end(), source->gas[g].label) !=
          seen.end())
        continue;
      seen.push_back(source->gas[g].label);
      if (p >= static_cast<unsigned>(SurfaceChemistryParamsGPU::maxParticles)) {
        VIENNACORE_LOG_ERROR("SurfaceChemistry GPU: too many traced species.");
        break;
      }

      viennaray::gpu::Particle<NumericType> particle{
          .name = source->gas[g].label,
          .sticking = static_cast<NumericType>(source->stickingOf(int(g)))};
      particle.dataLabels.push_back(source->gas[g].label);

      deviceParams_.freeSiteExponent[p] =
          source->gas[g].stickingFreeSiteExponent;
      deviceParams_.stickingSite[p] = source->gas[g].stickingSite;

      // per-material sticking, evaluated once here so the shader only looks up
      const auto &d = source->gas[g].stickingConstant.getDefault();
      deviceParams_.defaultSticking[p] =
          static_cast<float>(source->arrhenius(d.prefactor, d.Ea, d.beta));
      int nOverride = 0;
#define PS_GPU_STICKING(id, sym, cat, dens, cond, color)                       \
  if (source->gas[g].stickingConstant.has(BuiltInMaterial::sym)) {             \
    if (nOverride < SurfaceChemistryParamsGPU::maxMaterials) {               \
      deviceParams_.overrideMaterial[p][nOverride] =                           \
          Material(BuiltInMaterial::sym).legacyId();                           \
      deviceParams_.overrideSticking[p][nOverride] = static_cast<float>(       \
          source->stickingOf(int(g), Material(BuiltInMaterial::sym)));         \
      ++nOverride;                                                             \
    } else {                                                                   \
      VIENNACORE_LOG_ERROR(                                                    \
          "SurfaceChemistry GPU: too many per-material stickings.");         \
    }                                                                          \
  }
      BUILTIN_MATERIAL_LIST(PS_GPU_STICKING)
#undef PS_GPU_STICKING
      deviceParams_.numOverrides[p] = nOverride;

      pMap[source->gas[g].label] = p;
      cMap.push_back({p, viennaray::gpu::CallableSlot::COLLISION,
                      "__direct_callable__chemicalNeutralCollision"});
      cMap.push_back({p, viennaray::gpu::CallableSlot::REFLECTION,
                      "__direct_callable__chemicalNeutralReflection"});

      this->insertNextParticleType(particle);
      ++p;
    }

    // One ion, writing a yield-weighted flux into a channel per ion reaction,
    // exactly as impl::ChemicalIon does on the CPU. It comes from whichever
    // step of the cycle declares one: a plasma ALD process has ions in the
    // plasma step and none in the dose.
    const ::viennaps::ChemicalMechanism<NumericType> *ionMech = nullptr;
    for (const auto *source : mechanismsToTrace())
      if (source->ionSource.present && !source->ionYields.empty()) {
        ionMech = source;
        break;
      }
    if (ionMech) {
      const auto &src = ionMech->ionSource;
      constexpr NumericType toRad = M_PI / 180.;

      deviceParams_.meanEnergy = static_cast<float>(src.meanEnergy);
      deviceParams_.sigmaEnergy = static_cast<float>(src.sigmaEnergy);
      deviceParams_.inflectAngle = static_cast<float>(src.inflectAngle * toRad);
      deviceParams_.n_l = static_cast<float>(src.n_l);
      deviceParams_.minAngle = static_cast<float>(src.minAngle * toRad);
      deviceParams_.thetaRMin = static_cast<float>(src.thetaRMin * toRad);
      deviceParams_.thetaRMax = static_cast<float>(src.thetaRMax * toRad);

      viennaray::gpu::Particle<NumericType> ion;
      ion.name = "ChemicalIon";
      ion.sticking = 0.f;
      ion.cosineExponent = static_cast<NumericType>(src.exponent);

      NumericType minEth = std::numeric_limits<NumericType>::max();
      int nYield = 0;
      for (const auto &y : ionMech->ionYields) {
        if (nYield >= SurfaceChemistryParamsGPU::maxYields) {
          VIENNACORE_LOG_ERROR("SurfaceChemistry GPU: too many ion yields.");
          break;
        }
        ion.dataLabels.push_back(y.label);
        deviceParams_.yieldA[nYield] = static_cast<float>(y.A);
        deviceParams_.yieldEth[nYield] = static_cast<float>(y.Eth);
        deviceParams_.yieldB[nYield] = static_cast<float>(y.B);
        deviceParams_.yieldEnhanced[nYield] = y.enhanced ? 1 : 0;
        minEth = std::min(minEth, y.Eth);

        int nOverride = 0;
#define PS_GPU_YIELD(id, sym, cat, dens, cond, color)                          \
  if (y.materialA.has(BuiltInMaterial::sym) ||                                 \
      y.materialEth.has(BuiltInMaterial::sym)) {                               \
    if (nOverride < SurfaceChemistryParamsGPU::maxMaterials) {               \
      const auto mat = Material(BuiltInMaterial::sym);                         \
      deviceParams_.yieldOverrideMaterial[nYield][nOverride] = mat.legacyId(); \
      deviceParams_.yieldOverrideA[nYield][nOverride] =                        \
          static_cast<float>(y.materialA.get(mat));                            \
      deviceParams_.yieldOverrideEth[nYield][nOverride] =                      \
          static_cast<float>(y.materialEth.get(mat));                          \
      ++nOverride;                                                             \
    } else {                                                                   \
      VIENNACORE_LOG_ERROR(                                                    \
          "SurfaceChemistry GPU: too many per-material ion yields.");        \
    }                                                                          \
  }
        BUILTIN_MATERIAL_LIST(PS_GPU_YIELD)
#undef PS_GPU_YIELD
        deviceParams_.yieldNumOverrides[nYield] = nOverride;
        ++nYield;
      }
      deviceParams_.numYields = nYield;
      deviceParams_.minEth = static_cast<float>(minEth);

      pMap[ion.name] = p;
      cMap.push_back({p, viennaray::gpu::CallableSlot::COLLISION,
                      "__direct_callable__chemicalIonCollision"});
      cMap.push_back({p, viennaray::gpu::CallableSlot::REFLECTION,
                      "__direct_callable__chemicalIonReflection"});
      cMap.push_back({p, viennaray::gpu::CallableSlot::INIT,
                      "__direct_callable__chemicalIonInit"});
      this->insertNextParticleType(ion);
      ++p;
    }

    this->setParticleCallableMap(pMap, cMap);
    // the shader resolves the material under every hit, for both the sticking
    // of a neutral and the yield of the ion
    this->setUseMaterialIds(true);

    // the free-site exponents and coverage count reach the shader as customData
    this->processData.alloc(sizeof(SurfaceChemistryParamsGPU));
    this->processData.upload(&deviceParams_, 1);

    surfModel_ =
        SmartPointer<impl::ChemicalSurfaceModel<NumericType, D>>::New(mech_);
    // addMechanism rebuilds this, so an atomic layer process requested before
    // the last mechanism was added stays requested, and the integrator keeps
    // its accuracy setting.
    surfModel_->setPreserveCoverages(this->isALP);
    surfModel_->setMaxCoverageChange(maxCoverageChange_);
    auto velField = SmartPointer<DefaultVelocityField<NumericType, D>>::New();

    this->setSurfaceModel(surfModel_);
    this->setVelocityField(velField);
    this->setProcessName("SurfaceChemistry");
    this->hasGPU = true;

    this->processMetaData = mech_.toProcessMetaData();
  }
};
} // namespace gpu
#endif

/// CVD deposition driven by a reaction mechanism supplied as data.
template <typename NumericType, int D>
class SurfaceChemistry final : public ProcessModelCPU<NumericType, D> {
public:
  SurfaceChemistry() = default;

  explicit SurfaceChemistry(const ChemicalMechanism<NumericType> &m)
      : mech(m) {
    initializeModel();
  }

  void setMechanism(const ChemicalMechanism<NumericType> &m) {
    mech = m;
    phaseMechanisms_.clear();
    initializeModel();
  }

  // Register the chemistry that governs one step of an atomic layer cycle. A
  // PE-ALD cycle is written as two reaction files, one per half-cycle, and the
  // coverages of one step are the initial condition of the next -- so the two
  // must describe the same surface. That is checked here rather than left to
  // go wrong quietly: an index mismatch between the two coverage lists would
  // silently carry the wrong species across.
  void addMechanism(const std::string &name,
                    const ChemicalMechanism<NumericType> &m) {
    if (!phaseMechanisms_.empty()) {
      const auto &first = phaseMechanisms_.front().second;
      if (m.coverageNames != first.coverageNames) {
        VIENNACORE_LOG_ERROR(
            "Mechanism '" + name + "' declares different coverages than '" +
            phaseMechanisms_.front().first +
            "'. The steps of one cycle must describe the same surface.");
        return;
      }
    }
    phaseMechanisms_.emplace_back(name, m);
    if (phaseMechanisms_.size() == 1)
      mech = m;
    initializeModel();
  }

  // Select the chemistry and the flowing species for one step of the cycle.
  // Everything not flowing has its source flux set to zero, so its rate law
  // contributes nothing while the reaction stays in the mechanism -- which is
  // what a purge is, and what the other half-cycle's reactants are during a
  // dose.
  void setActivePhase(const std::string &phaseName,
                      const std::vector<std::string> &activeSpecies,
                      const std::string &mechanismName) override {
    if (!mechanismName.empty()) {
      const auto it = std::find_if(
          phaseMechanisms_.begin(), phaseMechanisms_.end(),
          [&](const auto &entry) { return entry.first == mechanismName; });
      if (it == phaseMechanisms_.end()) {
        VIENNACORE_LOG_ERROR("Phase '" + phaseName + "' names mechanism '" +
                             mechanismName + "', which was not registered.");
        return;
      }
      mech = it->second;
    }

    // "*" is a pulse that flows everything, which is what a process with one
    // reactant means by a pulse.
    const bool all = activeSpecies.size() == 1 && activeSpecies.front() == "*";
    for (auto &g : mech.gas) {
      if (g.label.empty())
        continue; // not traced, so it carries no source flux
      const bool flowing =
          all || std::find(activeSpecies.begin(), activeSpecies.end(),
                           g.label) != activeSpecies.end();
      g.sourceFlux = flowing ? sourceFluxOf(g.label) : NumericType(0.);
    }
  }

  ChemicalMechanism<NumericType> &getMechanism() { return mech; }

  // Growth per cycle on the open field, in the length unit of the process.
  // When the process ran on the device, the model that actually ran is the
  // converted one this class handed out, so the question is passed to it.
  NumericType growthPerCycle() const {
#ifdef VIENNACORE_COMPILE_GPU
    if (gpuModel_ && gpuModel_->growthPerCycle() > 0.)
      return gpuModel_->growthPerCycle();
#endif
    if (!surfModel_ || surfModel_->getCyclesRun() == 0)
      return 0.;
    return surfModel_->getFieldGrowth() / surfModel_->getCyclesRun();
  }

  // Run this mechanism as one step of an atomic layer process: coverages are
  // integrated in time rather than solved at steady state, and they survive
  // from one step of the cycle into the next. An ALD chemistry has no steady
  // state to solve for -- see ChemicalMechanism::stepCoverages -- so a cyclic
  // process must take this path.
  void setAtomicLayerProcess(bool enable = true) {
    this->isALP = enable;
    if (surfModel_)
      surfModel_->setPreserveCoverages(enable);
  }

  // The incident flux of one gas species, for gating a step of the cycle.
  void setSourceFlux(const std::string &label, NumericType flux) {
    if (!mech.setSourceFlux(label, flux))
      VIENNACORE_LOG_WARNING("No gas species '" + label +
                             "' in mechanism '" + mech.name + "'.");
  }

  // Accuracy of the transient coverage integration; see
  // ChemicalSurfaceModel::setMaxCoverageChange.
  void setMaxCoverageChange(NumericType maxChange) {
    if (surfModel_)
      surfModel_->setMaxCoverageChange(maxChange);
    maxCoverageChange_ = maxChange;
  }

#ifdef VIENNACORE_COMPILE_GPU
  SmartPointer<ProcessModelBase<NumericType, D>> getGPUModel() override {
    auto &model = gpuModel_;
    if (phaseMechanisms_.empty()) {
      model = SmartPointer<gpu::SurfaceChemistry<NumericType, D>>::New(mech);
    } else {
      // A cyclic process carries all of its steps across, not just the one
      // that happens to be active, or the device would hold the particles of
      // a single half-cycle.
      model = SmartPointer<gpu::SurfaceChemistry<NumericType, D>>::New();
      model->setAtomicLayerProcess(this->isALP);
      model->setMaxCoverageChange(maxCoverageChange_);
      for (const auto &entry : phaseMechanisms_)
        model->addMechanism(entry.first, entry.second);
    }
    model->setProcessName(this->getProcessName().value());
    return model;
  }
#endif

private:
  // Every mechanism whose species must be traced: the registered phases if a
  // cycle was built, otherwise the single mechanism this model was given.
  std::vector<const ChemicalMechanism<NumericType> *> mechanismsToTrace() const {
    std::vector<const ChemicalMechanism<NumericType> *> out;
    if (phaseMechanisms_.empty()) {
      out.push_back(&mech);
    } else {
      for (const auto &entry : phaseMechanisms_)
        out.push_back(&entry.second);
    }
    return out;
  }

  void initializeModel() {
    if (units::Length::getUnit() == units::Length::UNDEFINED ||
        units::Time::getUnit() == units::Time::UNDEFINED) {
      VIENNACORE_LOG_ERROR("Units have not been set.");
    }

    surfModel_ =
        SmartPointer<impl::ChemicalSurfaceModel<NumericType, D>>::New(mech);
    // setMechanism rebuilds the surface model, so an atomic layer process that
    // was requested before the mechanism was replaced stays requested, and the
    // integrator keeps its accuracy setting.
    surfModel_->setPreserveCoverages(this->isALP);
    surfModel_->setMaxCoverageChange(maxCoverageChange_);
    auto velField = SmartPointer<DefaultVelocityField<NumericType, D>>::New();

    this->setSurfaceModel(surfModel_);
    this->setVelocityField(velField);
    this->setProcessName("SurfaceChemistry");
    this->hasGPU = true;
    this->particles.clear();

    // One particle per traced species, over EVERY registered mechanism, not
    // just the active one. The particle set is fixed when the flux engine is
    // initialised, so it cannot change as the cycle moves from one step to the
    // next; a species that is not flowing is silenced by its source flux
    // instead. Each keeps the sticking of the file it came from.
    std::vector<std::string> seen;
    for (const auto &source : mechanismsToTrace()) {
      const int nCov = static_cast<int>(source->coverageNames.size());
      for (size_t g = 0; g < source->gas.size(); ++g) {
        const auto &gas = source->gas[g];
        if (!gas.traced || gas.isIonChannel)
          continue; // an ion channel is written by the ion, not by a neutral
        if (std::find(seen.begin(), seen.end(), gas.label) != seen.end())
          continue;
        seen.push_back(gas.label);
        // the coverages the particle's re-emission sees are those on the site
        // type its adsorption consumes free sites of
        std::vector<int> siteCov;
        for (int i = 0; i < nCov; ++i)
          if (source->coverageSite[i] == gas.stickingSite)
            siteCov.push_back(i);
        auto particle =
            std::make_unique<impl::ChemicalParticle<NumericType, D>>(
                gas.label, source->stickingTable(int(g)),
                gas.stickingFreeSiteExponent, std::move(siteCov));
        this->insertNextParticleType(particle);
      }
    }

    // One ion, writing a yield-weighted flux into a channel per ion reaction.
    // It comes from whichever step of the cycle declares one -- a plasma ALD
    // process has ions in the plasma step and none in the dose, and the ion
    // channels of the plasma chemistry would otherwise look for a flux no
    // particle ever wrote.
    for (const auto &source : mechanismsToTrace()) {
      if (source->ionSource.present && !source->ionYields.empty()) {
        auto ion = std::make_unique<impl::ChemicalIon<NumericType, D>>(
            source->ionSource, source->ionYields);
        this->insertNextParticleType(ion);
        break;
      }
    }

    this->processMetaData = mech.toProcessMetaData();
    this->processMetaData["Units"] = std::vector<double>{
        static_cast<double>(units::Length::getInstance().getUnit()),
        static_cast<double>(units::Time::getInstance().getUnit())};
  }

  // The flux a species carries in the file it came from, before any gating.
  NumericType sourceFluxOf(const std::string &label) const {
    for (const auto &entry : phaseMechanisms_)
      for (const auto &g : entry.second.gas)
        if (g.label == label)
          return g.sourceFlux;
    return NumericType(0.);
  }

  ChemicalMechanism<NumericType> mech; // the chemistry of the current phase
  std::vector<std::pair<std::string, ChemicalMechanism<NumericType>>>
      phaseMechanisms_;
  NumericType maxCoverageChange_ = 1e-3;
  SmartPointer<impl::ChemicalSurfaceModel<NumericType, D>> surfModel_ = nullptr;
#ifdef VIENNACORE_COMPILE_GPU
  SmartPointer<gpu::SurfaceChemistry<NumericType, D>> gpuModel_ = nullptr;
#endif
};

PS_PRECOMPILE_PRECISION_DIMENSION(SurfaceChemistry)

} // namespace viennaps
