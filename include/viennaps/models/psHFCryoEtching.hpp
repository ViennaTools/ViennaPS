#pragma once

#include <cmath>
#include <limits>
#include <vector>

#include <rayParticle.hpp>
#include <rayReflection.hpp>

#include "../process/psProcessModel.hpp"
#include "../psUnits.hpp"

#include "psHFCryoParameters.hpp"
#include "psIonModelUtil.hpp"

namespace viennaps {

using namespace viennacore;

// HF cryogenic plasma etching of SiO2.
//
// Two-state adsorption + surface diffusion model:
//
//   Physisorbed HF (theta_phys) — solved via 1D diffusion PDE along surface:
//     D_s * d²theta/ds² + Gamma_HF*(1-theta) - k_des*theta
//                       - k_act*Gamma_ion*theta = 0
//     BC: Dirichlet at chain ends (open surface = local Langmuir equilibrium)
//
//   Chemisorbed HF (theta_chem) — ion-activated, zero on sidewalls (Lill 2023):
//     theta_chem = k_act * Gamma_ion * theta_phys / k_r   (bottom/non-sidewall)
//     theta_chem = 0                                       (sidewall, per paper)
//
//   Etch rate:
//     v = -(Y_ie * Gamma_ion * theta_phys
//         + k_r  * theta_chem
//         + Y_sp * Gamma_ion) / rho

namespace impl {

// ── Ion ──────────────────────────────────────────────────────────────────────
template <typename NumericType, int D>
class HFCryoIon final
    : public viennaray::Particle<HFCryoIon<NumericType, D>, NumericType> {
public:
  explicit HFCryoIon(const HFCryoParameters<NumericType> &pParams)
      : params(pParams),
        A_energy(1. / (1. + params.Ions.n_l *
                                (M_PI_2 / params.Ions.inflectAngle - 1.))),
        sqrt_E_th_ie(std::sqrt(params.SiO2.Eth_ie)) {}

  void surfaceCollision(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                        const Vec3D<NumericType> &geomNormal,
                        const unsigned int primID, const int,
                        viennaray::TracingData<NumericType> &localData,
                        const viennaray::TracingData<NumericType> *,
                        RNG &) override {
    const NumericType cosTheta = -DotProduct(rayDir, geomNormal);
    const NumericType angle =
        std::acos(std::max(std::min(cosTheta, NumericType(1)), NumericType(0)));

    NumericType f_ie = 1.;
    if (cosTheta < 0.5)
      f_ie = std::max(NumericType(3) - NumericType(6) * angle / NumericType(M_PI),
                      NumericType(0));

    const NumericType sqrtE = std::sqrt(E);

    const NumericType Y_sp =
        params.SiO2.A_sp *
        std::max(sqrtE - std::sqrt(params.SiO2.Eth_sp), NumericType(0));

    const NumericType Y_ie =
        params.SiO2.A_ie *
        std::max(sqrtE - sqrt_E_th_ie, NumericType(0)) * f_ie;

    localData.getVectorData(0)[primID] += Y_sp * rayWeight;
    localData.getVectorData(1)[primID] += Y_ie * rayWeight;
  }

  std::pair<NumericType, Vec3D<NumericType>>
  surfaceReflection(NumericType, const Vec3D<NumericType> &rayDir,
                    const Vec3D<NumericType> &geomNormal,
                    const unsigned int, const int,
                    const viennaray::TracingData<NumericType> *,
                    RNG &rng) override {
    const NumericType cosTheta = -DotProduct(rayDir, geomNormal);
    const NumericType incAngle =
        std::acos(std::max(std::min(cosTheta, NumericType(1)), NumericType(0)));

    const NumericType newEnergy = updateEnergy(
        rng, E, incAngle, A_energy, params.Ions.inflectAngle, params.Ions.n_l);

    if (newEnergy > params.SiO2.Eth_ie) {
      E = newEnergy;
      auto dir = viennaray::ReflectionConedCosine<NumericType, D>(
          rayDir, geomNormal, rng,
          M_PI_2 - std::min(incAngle, params.Ions.minAngle));
      return {NumericType(0), dir};
    }
    return VIENNARAY_PARTICLE_STOP;
  }

  void initNew(RNG &rng) override {
    E = initNormalDistEnergy(rng, params.Ions.meanEnergy, params.Ions.sigmaEnergy);
  }

  NumericType getSourceDistributionPower() const override {
    return params.Ions.exponent;
  }

  [[nodiscard]] std::vector<std::string> getLocalDataLabels() const override {
    return {"ionSputterFlux", "ionActivationFlux"};
  }

private:
  const HFCryoParameters<NumericType> &params;
  const NumericType A_energy;
  const NumericType sqrt_E_th_ie;
  NumericType E = 0.;
};

// ── HF etchant ───────────────────────────────────────────────────────────────
template <typename NumericType, int D>
class HFCryoEtchant final
    : public viennaray::Particle<HFCryoEtchant<NumericType, D>, NumericType> {
public:
  explicit HFCryoEtchant(const HFCryoParameters<NumericType> &pParams)
      : params(pParams) {}

  void surfaceCollision(NumericType rayWeight, const Vec3D<NumericType> &,
                        const Vec3D<NumericType> &, const unsigned int primID,
                        const int,
                        viennaray::TracingData<NumericType> &localData,
                        const viennaray::TracingData<NumericType> *globalData,
                        RNG &) override {
    const NumericType theta_phys = globalData->getVectorData(0)[primID];
    const NumericType theta_chem = globalData->getVectorData(1)[primID];
    const NumericType occupied =
        std::min(theta_phys + theta_chem, NumericType(1));
    const NumericType S_eff =
        params.gamma_HF * std::max(NumericType(1) - occupied, NumericType(0));
    localData.getVectorData(0)[primID] += rayWeight * S_eff;
  }

  std::pair<NumericType, Vec3D<NumericType>>
  surfaceReflection(NumericType, const Vec3D<NumericType> &,
                    const Vec3D<NumericType> &geomNormal,
                    const unsigned int primID, const int,
                    const viennaray::TracingData<NumericType> *globalData,
                    RNG &rng) override {
    const NumericType theta_phys = globalData->getVectorData(0)[primID];
    const NumericType theta_chem = globalData->getVectorData(1)[primID];
    const NumericType occupied =
        std::min(theta_phys + theta_chem, NumericType(1));
    const NumericType S_eff =
        params.gamma_HF * std::max(NumericType(1) - occupied, NumericType(0));
    auto dir = viennaray::ReflectionDiffuse<NumericType, D>(geomNormal, rng);
    return {NumericType(1) - S_eff, dir};  // reflected weight = 1 - sticking prob
  }

  NumericType getSourceDistributionPower() const override { return 1.; }

  [[nodiscard]] std::vector<std::string> getLocalDataLabels() const override {
    return {"etchantFlux"};
  }

private:
  const HFCryoParameters<NumericType> &params;
};

// ── Surface model ─────────────────────────────────────────────────────────────
template <typename NumericType, int D>
class HFCryoSurfaceModel : public SurfaceModel<NumericType> {
public:
  using SurfaceModel<NumericType>::coverages;
  using SurfaceModel<NumericType>::surfaceData;

  explicit HFCryoSurfaceModel(const HFCryoParameters<NumericType> &pParams)
      : params(pParams) {}

  void initializeCoverages(unsigned numGeometryPoints) override {
    if (coverages == nullptr)
      coverages = viennals::PointData<NumericType>::New();
    else
      coverages->clear();

    std::vector<NumericType> zeros(numGeometryPoints, 0.);
    coverages->insertNextScalarData(zeros, "physCoverage"); // index 0
    coverages->insertNextScalarData(zeros, "chemCoverage"); // index 1
  }

  void initializeSurfaceData(unsigned numGeometryPoints) override {
    if (!Logger::hasIntermediate())
      return;
    if (surfaceData == nullptr)
      surfaceData = viennals::PointData<NumericType>::New();
    else
      surfaceData->clear();

    std::vector<NumericType> zeros(numGeometryPoints, 0.);
    surfaceData->insertNextScalarData(zeros, "chemicalRate");
    surfaceData->insertNextScalarData(zeros, "sputterRate");
  }

  // Local (point-wise) steady-state — no diffusion.
  // Called first; result is used as Dirichlet BC at chain endpoints.
  void updateCoverages(SmartPointer<viennals::PointData<NumericType>> rates,
                       const std::vector<NumericType> &) override {
    const auto numPoints   = rates->getScalarData(0)->size();
    const auto etchantFlux = rates->getScalarData("etchantFlux");
    const auto ionActFlux  = rates->getScalarData("ionActivationFlux");

    auto theta_phys = coverages->getScalarData("physCoverage");
    auto theta_chem = coverages->getScalarData("chemCoverage");
    theta_phys->resize(numPoints);
    theta_chem->resize(numPoints);

    const NumericType k_des      = params.effective_k_des();
    const NumericType k_r        = params.effective_k_r();
    const NumericType k_r_direct = params.effective_k_r_direct();
    const NumericType A_act      = params.IonActivation.A_act;

#pragma omp parallel for
    for (size_t i = 0; i < numPoints; ++i) {
      const NumericType Gamma_HF = etchantFlux->at(i) * params.etchantFlux;

      if (!params.Config.usePhysisorption) {
        // Single coverage: no chemisorption distinction
        const NumericType denom = Gamma_HF + k_des;
        theta_phys->at(i) = (denom > NumericType(1e-30))
                                ? std::min(Gamma_HF / denom, NumericType(1))
                                : NumericType(0);
        theta_chem->at(i) = NumericType(0);
      } else {
        // Two-state: physisorbed + ion-activated chemisorbed
        const NumericType k_act_ion = ionActFlux->at(i) * params.ionFlux * A_act;
        const NumericType denom = Gamma_HF + k_des + k_act_ion + k_r_direct;
        theta_phys->at(i) = (denom > NumericType(1e-30))
                                ? std::min(Gamma_HF / denom, NumericType(1))
                                : NumericType(0);
        const NumericType num_chem = k_act_ion * theta_phys->at(i);
        theta_chem->at(i) = (k_r > NumericType(1e-30))
                                ? std::min(num_chem / k_r,
                                           NumericType(1) - theta_phys->at(i))
                                : NumericType(0);
      }
    }
  }

  SmartPointer<std::vector<NumericType>>
  calculateVelocities(SmartPointer<viennals::PointData<NumericType>> rates,
                      const std::vector<Vec3D<NumericType>> &coordinates,
                      const std::vector<NumericType> &materialIds) override {
    // Step 1: local coverage (no diffusion) — also serves as Dirichlet BC values
    updateCoverages(rates, materialIds);

    const auto numPoints = rates->getScalarData(0)->size();

    const auto etchantFlux = rates->getScalarData("etchantFlux");
    const auto ionActFlux  = rates->getScalarData("ionActivationFlux");

    const NumericType k_des      = params.effective_k_des();
    const NumericType k_r        = params.effective_k_r();
    const NumericType k_r_direct = params.effective_k_r_direct();
    const NumericType A_act = params.IonActivation.A_act;
    const NumericType Ds    = params.D_s();

    std::vector<NumericType> Gamma_HF_vec(numPoints);
    std::vector<NumericType> k_act_ion_vec(numPoints);
    for (size_t i = 0; i < numPoints; ++i) {
      Gamma_HF_vec[i]  = etchantFlux->at(i) * params.etchantFlux;
      k_act_ion_vec[i] = ionActFlux->at(i) * params.ionFlux * A_act;
    }

    auto theta_phys = coverages->getScalarData("physCoverage");
    auto theta_chem = coverages->getScalarData("chemCoverage");

    // Step 2: build surface chain + sidewall flags
    const auto chain    = buildSurfaceChain(coordinates);
    const auto sidewall = buildSidewallFlags(chain, coordinates);

    // Step 3: surface diffusion (only when enabled and D0 > 0)
    if (params.Config.useSurfaceDiffusion && Ds > NumericType(0))
      applySurfaceDiffusion(*theta_phys, Gamma_HF_vec, k_act_ion_vec,
                            chain, coordinates, Ds, k_des, k_r_direct);

    // Step 4: recompute theta_chem
    if (params.Config.usePhysisorption) {
      // Two-state: zero on sidewalls (Lill 2023), steady-state elsewhere
      for (size_t i = 0; i < numPoints; ++i) {
        if (sidewall[i]) {
          theta_chem->at(i) = NumericType(0);
          continue;
        }
        const NumericType num_chem = k_act_ion_vec[i] * theta_phys->at(i);
        theta_chem->at(i) = (k_r > NumericType(1e-30))
                                ? std::min(num_chem / k_r,
                                           NumericType(1) - theta_phys->at(i))
                                : NumericType(0);
      }
    } else {
      // Single coverage: no chemisorption
      std::fill(theta_chem->begin(), theta_chem->end(), NumericType(0));
    }

    // Step 5: etch rate
    const auto ionSputterFlux    = rates->getScalarData("ionSputterFlux");
    const auto ionActivationFlux = rates->getScalarData("ionActivationFlux");
    const NumericType unitConversion =
        units::Time::convertSecond() / units::Length::convertNanometer();

    std::vector<NumericType> etchRate(numPoints, 0.);

    std::vector<NumericType> *chRate = nullptr, *spRate = nullptr;
    if (Logger::hasIntermediate()) {
      chRate = surfaceData->getScalarData("chemicalRate");
      spRate = surfaceData->getScalarData("sputterRate");
      chRate->resize(numPoints);
      spRate->resize(numPoints);
    }

    bool stop = false;
#pragma omp parallel for reduction(|| : stop)
    for (size_t i = 0; i < numPoints; ++i) {
      if (coordinates[i][D - 1] < params.etchStopDepth || stop) {
        stop = true;
        continue;
      }
      if (!MaterialMap::isMaterial(materialIds[i], Material::SiO2))
        continue;

      // Ion-enhanced etching: Y_ie * Gamma_ion * theta_phys (or single theta)
      // Pure chemical: k_r_direct * theta_phys (only when usePhysisorption,
      //   otherwise the concept of "physisorbed" HF is not defined)
      const NumericType ionEnhanced  =
          ionActivationFlux->at(i) * params.ionFlux * theta_phys->at(i);
      const NumericType pureChemical = k_r_direct * theta_phys->at(i);
      const NumericType sputter      = ionSputterFlux->at(i) * params.ionFlux;

      etchRate[i] = -(1. / params.SiO2.rho) *
                    (ionEnhanced + pureChemical + sputter) *
                    unitConversion;

      if (Logger::hasIntermediate()) {
        chRate->at(i) = ionEnhanced + pureChemical;
        spRate->at(i) = sputter;
      }
    }

    if (stop) {
      std::fill(etchRate.begin(), etchRate.end(), NumericType(0));
      VIENNACORE_LOG_INFO("Etch stop depth reached.");
    }

    return SmartPointer<std::vector<NumericType>>::New(std::move(etchRate));
  }

private:
  const HFCryoParameters<NumericType> &params;

  // Nearest-neighbor chain ordering surface points as a 1D curve.
  // Starts from the topmost point (open to plasma).
  std::vector<size_t>
  buildSurfaceChain(const std::vector<Vec3D<NumericType>> &points) const {
    const size_t N = points.size();
    if (N == 0) return {};

    std::vector<bool> visited(N, false);
    std::vector<size_t> chain;
    chain.reserve(N);

    size_t start = 0;
    for (size_t i = 1; i < N; ++i)
      if (points[i][D - 1] > points[start][D - 1])
        start = i;

    chain.push_back(start);
    visited[start] = true;

    NumericType thresholdSq = NumericType(9);

    for (size_t step = 1; step < N; ++step) {
      const size_t curr = chain.back();
      NumericType bestDistSq = std::numeric_limits<NumericType>::max();
      size_t best = N;

      for (size_t j = 0; j < N; ++j) {
        if (visited[j]) continue;
        NumericType distSq = NumericType(0);
        for (int d = 0; d < D; ++d) {
          NumericType dd = points[j][d] - points[curr][d];
          distSq += dd * dd;
        }
        if (distSq < bestDistSq) {
          bestDistSq = distSq;
          best = j;
        }
      }

      if (best == N || bestDistSq > thresholdSq)
        break;

      if (step == 1)
        thresholdSq = bestDistSq * NumericType(4);

      chain.push_back(best);
      visited[best] = true;
    }

    return chain;
  }

  // Classify each point in chain as sidewall (true) or not (false).
  // Sidewall: local chain tangent is predominantly in height direction (D-1).
  std::vector<bool>
  buildSidewallFlags(const std::vector<size_t> &chain,
                     const std::vector<Vec3D<NumericType>> &points) const {
    const size_t numPoints = points.size();
    const size_t M = chain.size();
    std::vector<bool> sidewall(numPoints, false);

    for (size_t j = 0; j < M; ++j) {
      // Centered tangent (or one-sided at endpoints)
      Vec3D<NumericType> tang{0, 0, 0};
      if (j == 0)
        for (int d = 0; d < D; ++d)
          tang[d] = points[chain[1]][d] - points[chain[0]][d];
      else if (j == M - 1)
        for (int d = 0; d < D; ++d)
          tang[d] = points[chain[M-1]][d] - points[chain[M-2]][d];
      else
        for (int d = 0; d < D; ++d)
          tang[d] = points[chain[j+1]][d] - points[chain[j-1]][d];

      NumericType len = NumericType(0);
      for (int d = 0; d < D; ++d) len += tang[d] * tang[d];
      len = std::sqrt(len);
      if (len < NumericType(1e-10)) continue;

      // Sidewall: tangent mostly vertical (height component > 0.7)
      const NumericType vertFrac = std::abs(tang[D - 1]) / len;
      sidewall[chain[j]] = (vertFrac > NumericType(0.7));
    }

    return sidewall;
  }

  // Solve steady-state surface diffusion PDE on 1D chain via Thomas algorithm.
  //
  // Boundary conditions (Dirichlet):
  //   theta[0]   = local Langmuir value at open-surface end (chain start)
  //   theta[M-1] = local Langmuir value at open-surface end (chain end)
  //
  // Interior (j = 1..M-2):
  //   -alpha*theta[j-1] + (2*alpha + k_des + Gamma[j] + k_act[j])*theta[j]
  //   - alpha*theta[j+1] = Gamma[j]
  void applySurfaceDiffusion(
      std::vector<NumericType> &theta_phys,
      const std::vector<NumericType> &Gamma_HF_vec,
      const std::vector<NumericType> &k_act_ion_vec,
      const std::vector<size_t> &chain,
      const std::vector<Vec3D<NumericType>> &points,
      NumericType Ds, NumericType k_des, NumericType k_r_direct) const {
    const size_t M = chain.size();
    if (M < 3) return;

    // Average arc-length spacing
    NumericType avgDs = NumericType(0);
    for (size_t j = 0; j + 1 < M; ++j) {
      NumericType distSq = NumericType(0);
      for (int d = 0; d < D; ++d) {
        NumericType dd = points[chain[j + 1]][d] - points[chain[j]][d];
        distSq += dd * dd;
      }
      avgDs += std::sqrt(distSq);
    }
    avgDs /= NumericType(M - 1);
    if (avgDs < NumericType(1e-10)) return;

    const NumericType alpha = Ds / (avgDs * avgDs);

    // Save Dirichlet values at chain endpoints (local Langmuir equilibrium)
    const NumericType bc_left  = theta_phys[chain[0]];
    const NumericType bc_right = theta_phys[chain[M - 1]];

    // Build tridiagonal system
    std::vector<NumericType> a(M, -alpha);
    std::vector<NumericType> b(M);
    std::vector<NumericType> c(M, -alpha);
    std::vector<NumericType> rhs(M);

    // Dirichlet BC at endpoints
    a[0]     = NumericType(0); b[0]     = NumericType(1);
    c[0]     = NumericType(0); rhs[0]   = bc_left;
    a[M - 1] = NumericType(0); b[M - 1] = NumericType(1);
    c[M - 1] = NumericType(0); rhs[M - 1] = bc_right;

    // Interior nodes: all theta_phys loss terms in diagonal
    for (size_t j = 1; j + 1 < M; ++j) {
      const size_t idx    = chain[j];
      const NumericType G = Gamma_HF_vec[idx];
      const NumericType K = k_act_ion_vec[idx];
      b[j]   = NumericType(2) * alpha + k_des + G + K + k_r_direct;
      rhs[j] = G;
    }

    // Thomas algorithm — forward sweep
    std::vector<NumericType> c_prime(M), d_prime(M);
    c_prime[0] = c[0] / b[0];
    d_prime[0] = rhs[0] / b[0];

    for (size_t j = 1; j < M; ++j) {
      const NumericType denom = b[j] - a[j] * c_prime[j - 1];
      const NumericType inv = (std::abs(denom) > NumericType(1e-30))
                                  ? NumericType(1) / denom
                                  : NumericType(0);
      c_prime[j] = c[j] * inv;
      d_prime[j] = (rhs[j] - a[j] * d_prime[j - 1]) * inv;
    }

    // Back substitution
    theta_phys[chain[M - 1]] =
        std::max(NumericType(0), std::min(d_prime[M - 1], NumericType(1)));

    for (size_t j = M - 2; ; --j) {
      theta_phys[chain[j]] =
          std::max(NumericType(0),
                   std::min(d_prime[j] - c_prime[j] * theta_phys[chain[j + 1]],
                            NumericType(1)));
      if (j == 0) break;
    }
  }
};

} // namespace impl

// ── Public process model ──────────────────────────────────────────────────────
template <typename NumericType, int D>
class HFCryoEtching final : public ProcessModelCPU<NumericType, D> {
public:
  HFCryoEtching() { initializeModel(); }

  explicit HFCryoEtching(const HFCryoParameters<NumericType> &pParams)
      : params(pParams) {
    initializeModel();
  }

  void setParameters(const HFCryoParameters<NumericType> &pParams) {
    params = pParams;
    initializeModel();
  }

  HFCryoParameters<NumericType> &getParameters() { return params; }

private:
  void initializeModel() {
    if (units::Length::getUnit() == units::Length::UNDEFINED ||
        units::Time::getUnit() == units::Time::UNDEFINED) {
      VIENNACORE_LOG_ERROR("Units have not been set.");
    }

    auto ion     = std::make_unique<impl::HFCryoIon<NumericType, D>>(params);
    auto etchant = std::make_unique<impl::HFCryoEtchant<NumericType, D>>(params);
    auto surfModel =
        SmartPointer<impl::HFCryoSurfaceModel<NumericType, D>>::New(params);
    auto velField =
        SmartPointer<DefaultVelocityField<NumericType, D>>::New();

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("HFCryoEtching");
    this->particles.clear();
    this->insertNextParticleType(ion);
    this->insertNextParticleType(etchant);
  }

  HFCryoParameters<NumericType> params;
};

PS_PRECOMPILE_PRECISION_DIMENSION(HFCryoEtching)

} // namespace viennaps
