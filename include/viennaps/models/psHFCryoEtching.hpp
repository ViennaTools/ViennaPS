#pragma once

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
// Physics:
//   SiO2 + 4HF -> SiF4(g) + 2H2O
//
// Coverage (steady-state):
//   phi_HF = Gamma_HF / (Gamma_HF + k_des(T) + k_r(T) + Y_ie * Gamma_ion)
//
//   k_des(T) = nu0  * exp(-E_des / kB*T)   [Frenkel-Arrhenius desorption]
//   k_r(T)   = A_r  * exp(-E_a  / kB*T)   [Arrhenius reaction rate]
//
// Etch rate:
//   v = -(1/rho) * (k_r(T)*phi_HF + Y_sp*Gamma_ion + Y_ie*Gamma_ion*phi_HF)
//
// No passivation layer is needed: anisotropy is provided by ion bombardment
// and temperature-suppressed chemical etching on sidewalls.

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
                        const unsigned int primID, const int /*materialId*/,
                        viennaray::TracingData<NumericType> &localData,
                        const viennaray::TracingData<NumericType> *,
                        RNG &) override {
    const NumericType cosTheta =
        -DotProduct(rayDir, geomNormal);
    const NumericType angle =
        std::acos(std::max(std::min(cosTheta, NumericType(1)), NumericType(0)));

    // Angular factor: ion-enhanced etching drops off at glancing angles
    NumericType f_ie = 1.;
    if (cosTheta < 0.5)
      f_ie = std::max(NumericType(3) - NumericType(6) * angle / NumericType(M_PI),
                      NumericType(0));

    const NumericType sqrtE = std::sqrt(E);

    // Physical sputtering yield
    const NumericType Y_sp =
        params.SiO2.A_sp *
        std::max(sqrtE - std::sqrt(params.SiO2.Eth_sp), NumericType(0));

    // Ion-enhanced chemical etching yield
    const NumericType Y_ie =
        params.SiO2.A_ie *
        std::max(sqrtE - sqrt_E_th_ie, NumericType(0)) * f_ie;

    // ionSputterFlux
    localData.getVectorData(0)[primID] += Y_sp * rayWeight;
    // ionEnhancedFlux
    localData.getVectorData(1)[primID] += Y_ie * rayWeight;
  }

  std::pair<NumericType, Vec3D<NumericType>>
  surfaceReflection(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                    const Vec3D<NumericType> &geomNormal,
                    const unsigned int /*primId*/, const int /*materialId*/,
                    const viennaray::TracingData<NumericType> *,
                    RNG &rng) override {
    const NumericType cosTheta =
        -DotProduct(rayDir, geomNormal);
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
    E = initNormalDistEnergy(rng, params.Ions.meanEnergy,
                             params.Ions.sigmaEnergy);
  }

  NumericType getSourceDistributionPower() const override {
    return params.Ions.exponent;
  }

  [[nodiscard]] std::vector<std::string> getLocalDataLabels() const override {
    return {"ionSputterFlux", "ionEnhancedFlux"};
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
                        const int /*materialId*/,
                        viennaray::TracingData<NumericType> &localData,
                        const viennaray::TracingData<NumericType> *globalData,
                        RNG &) override {
    // Effective sticking: HF only adsorbs on uncovered surface
    const NumericType phi_HF = globalData->getVectorData(0)[primID];
    const NumericType S_eff =
        params.gamma_HF * std::max(NumericType(1) - phi_HF, NumericType(0));

    localData.getVectorData(0)[primID] += rayWeight * S_eff;
  }

  std::pair<NumericType, Vec3D<NumericType>>
  surfaceReflection(NumericType /*rayWeight*/, const Vec3D<NumericType> &,
                    const Vec3D<NumericType> &geomNormal,
                    const unsigned int primID, const int /*materialId*/,
                    const viennaray::TracingData<NumericType> *globalData,
                    RNG &rng) override {
    const NumericType phi_HF = globalData->getVectorData(0)[primID];
    const NumericType S_eff =
        params.gamma_HF * std::max(NumericType(1) - phi_HF, NumericType(0));

    auto dir = viennaray::ReflectionDiffuse<NumericType, D>(geomNormal, rng);
    return {S_eff, dir};
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
    coverages->insertNextScalarData(zeros, "HFCoverage");
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
    surfaceData->insertNextScalarData(zeros, "ionEnhancedRate");
    surfaceData->insertNextScalarData(zeros, "sputterRate");
  }

  SmartPointer<std::vector<NumericType>>
  calculateVelocities(SmartPointer<viennals::PointData<NumericType>> rates,
                      const std::vector<Vec3D<NumericType>> &coordinates,
                      const std::vector<NumericType> &materialIds) override {
    updateCoverages(rates, materialIds);

    const auto numPoints = rates->getScalarData(0)->size();
    std::vector<NumericType> etchRate(numPoints, 0.);

    const auto ionSputterFlux = rates->getScalarData("ionSputterFlux");
    const auto ionEnhancedFlux = rates->getScalarData("ionEnhancedFlux");
    const auto phi_HF = coverages->getScalarData("HFCoverage");

    std::vector<NumericType> *chRate = nullptr, *ieRate = nullptr,
                             *spRate = nullptr;
    if (Logger::hasIntermediate()) {
      chRate = surfaceData->getScalarData("chemicalRate");
      ieRate = surfaceData->getScalarData("ionEnhancedRate");
      spRate = surfaceData->getScalarData("sputterRate");
      chRate->resize(numPoints);
      ieRate->resize(numPoints);
      spRate->resize(numPoints);
    }

    const NumericType kr = params.k_r();
    const NumericType unitConversion =
        units::Time::convertSecond() / units::Length::convertNanometer();

    bool stop = false;

#pragma omp parallel for reduction(|| : stop)
    for (size_t i = 0; i < numPoints; ++i) {
      if (coordinates[i][D - 1] < params.etchStopDepth || stop) {
        stop = true;
        continue;
      }

      if (!MaterialMap::isMaterial(materialIds[i], Material::SiO2))
        continue;

      const NumericType chemical =
          kr * phi_HF->at(i);
      const NumericType ionEnhanced =
          phi_HF->at(i) * ionEnhancedFlux->at(i) * params.ionFlux;
      const NumericType sputter =
          ionSputterFlux->at(i) * params.ionFlux;

      etchRate[i] = -(1. / params.SiO2.rho) *
                    (chemical + ionEnhanced + sputter) * unitConversion;

      if (Logger::hasIntermediate()) {
        chRate->at(i) = chemical;
        ieRate->at(i) = ionEnhanced;
        spRate->at(i) = sputter;
      }
    }

    if (stop) {
      std::fill(etchRate.begin(), etchRate.end(), NumericType(0));
      VIENNACORE_LOG_INFO("Etch stop depth reached.");
    }

    return SmartPointer<std::vector<NumericType>>::New(std::move(etchRate));
  }

  void updateCoverages(SmartPointer<viennals::PointData<NumericType>> rates,
                       const std::vector<NumericType> & /*materialIds*/) override {
    const auto numPoints = rates->getScalarData(0)->size();
    const auto etchantFlux = rates->getScalarData("etchantFlux");
    const auto ionEnhancedFlux = rates->getScalarData("ionEnhancedFlux");

    auto phi_HF = coverages->getScalarData("HFCoverage");
    phi_HF->resize(numPoints);

    const NumericType k_des = params.k_des();
    const NumericType k_r = params.k_r();

#pragma omp parallel for
    for (size_t i = 0; i < numPoints; ++i) {
      // Adsorption flux (scaled)
      const NumericType Gamma_HF = etchantFlux->at(i) * params.etchantFlux;
      // Ion-enhanced removal of adsorbed HF
      const NumericType Y_ie_ion =
          ionEnhancedFlux->at(i) * params.ionFlux;

      // Steady-state coverage:
      //   phi = Gamma_HF / (Gamma_HF + k_des + k_r + Y_ie_ion)
      const NumericType denom = Gamma_HF + k_des + k_r + Y_ie_ion;
      phi_HF->at(i) = (denom > NumericType(1e-30))
                          ? std::min(Gamma_HF / denom, NumericType(1))
                          : NumericType(0);
    }
  }

private:
  const HFCryoParameters<NumericType> &params;
};

} // namespace impl

// ── Public process model ──────────────────────────────────────────────────────
template <typename NumericType, int D>
class HFCryoEtching final : public ProcessModelCPU<NumericType, D> {
public:
  HFCryoEtching() { initializeModel(); }

  // Convenience constructor
  HFCryoEtching(NumericType ionFlux, NumericType etchantFlux,
                NumericType temperature,   // K
                NumericType meanEnergy,    // eV
                NumericType sigmaEnergy,   // eV
                NumericType E_des = 0.25,  // eV  (Frenkel desorption)
                NumericType E_a = 0.10,    // eV  (reaction activation)
                NumericType etchStopDepth =
                    std::numeric_limits<NumericType>::lowest()) {
    params.ionFlux = ionFlux;
    params.etchantFlux = etchantFlux;
    params.temperature = temperature;
    params.Ions.meanEnergy = meanEnergy;
    params.Ions.sigmaEnergy = sigmaEnergy;
    params.Desorption.E_des = E_des;
    params.Reaction.E_a = E_a;
    params.etchStopDepth = etchStopDepth;
    initializeModel();
  }

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

    auto ion = std::make_unique<impl::HFCryoIon<NumericType, D>>(params);
    auto etchant =
        std::make_unique<impl::HFCryoEtchant<NumericType, D>>(params);

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
