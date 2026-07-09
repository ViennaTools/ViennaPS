#pragma once

#include <rayParticle.hpp>

#include "../process/psProcessModel.hpp"
#include "../process/psSurfaceModel.hpp"
#include "../process/psVelocityField.hpp"
#include "../psConstants.hpp"
#include "../psUnits.hpp"

#include "psPlasmaEtching.hpp"
#include "psPlasmaEtchingParameters.hpp"

namespace viennaps {

using namespace viennacore;

// Dedicated CF4/Ar silicon etching model.
//
// This is a thin configuration of the generic ViennaPS PlasmaEtching framework
// (PlasmaEtchingParameters + PlasmaEtchingSurfaceModel / PlasmaEtchingIon /
// PlasmaEtchingNeutral), the same class used by SF6O2Etching and HBrO2Etching.
// Because it reuses that framework it automatically supports GPU ray tracing
// via getGPUModel() and the generic plasma OptiX shaders.
//
// Species / surface-state mapping for the CF4/Ar-on-Si benchmark:
//   Ion         -> Ar+ ion bombardment
//   Etchant     -> F  neutral radical, coverage eCoverage == theta_F
//   Passivation -> lumped CFx radical,  coverage pCoverage == theta_CF
//   Substrate   -> Si
//
//   k_sigma            : chemical (F) etch coefficient  (k_chem)
//   Passivation.A_ie   : Ar+ removal yield of CFx residue
//   beta_E / beta_P    : F / CFx sticking probabilities
//
// The staged chemistry from the benchmark plan follows directly:
//   Step 1 (F + Ar+):  passivationFlux == 0  -> only theta_F is active.
//   Step 2 (+ CFx):    passivationFlux  > 0  -> theta_CF passivates the
//                      surface and blocks F adsorption.

#ifdef VIENNACORE_COMPILE_GPU
namespace gpu {
/// GPU version of the CF4/Ar plasma etching model
template <typename NumericType, int D>
class CF4ArEtching final : public ProcessModelGPU<NumericType, D> {
public:
  explicit CF4ArEtching(const PlasmaEtchingParameters<NumericType> &pParams)
      : params(pParams), deviceParams(pParams) {
    initializeModel();
  }

  ~CF4ArEtching() override { this->processData.free(); }

private:
  void initializeModel() {
    // particles
    viennaray::gpu::Particle<NumericType> ion;
    ion.name = "Ion"; // name for shader programs postfix
    ion.dataLabels.push_back("ionSputterFlux");
    ion.dataLabels.push_back("ionEnhancedFlux");
    ion.dataLabels.push_back("ionEnhancedPassivationFlux");
    ion.sticking = 0.f;
    ion.cosineExponent = params.Ions.exponent;

    viennaray::gpu::Particle<NumericType> fluorine;
    fluorine.name = "Etchant";
    fluorine.dataLabels.push_back("etchantFlux");
    fluorine.cosineExponent = 1.f;
    for (auto entry : params.beta_E) {
      fluorine.materialSticking[static_cast<int>(entry.material)] = entry.value;
    }

    viennaray::gpu::Particle<NumericType> cfx;
    cfx.name = "Oxygen"; // lumped CFx channel (generic passivation neutral)
    cfx.dataLabels.push_back("passivationFlux");
    cfx.cosineExponent = 1.f;
    for (auto entry : params.beta_P) {
      cfx.materialSticking[static_cast<int>(entry.material)] = entry.value;
    }

    // surface model
    auto surfModel = SmartPointer<
        viennaps::impl::PlasmaEtchingSurfaceModel<NumericType, D>>::New(params);

    // velocity field
    auto velField = SmartPointer<DefaultVelocityField<NumericType, D>>::New();

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("CF4ArEtching");
    this->getParticleTypes().clear();
    this->hasGPU = true;

    this->insertNextParticleType(ion);
    this->insertNextParticleType(fluorine);
    this->insertNextParticleType(cfx);

    std::unordered_map<std::string, unsigned> pMap = {
        {"Ion", 0}, {"Etchant", 1}, {"Oxygen", 2}};
    std::vector<viennaray::gpu::CallableConfig> cMap = {
        {0, viennaray::gpu::CallableSlot::COLLISION,
         "__direct_callable__plasmaIonCollision"},
        {0, viennaray::gpu::CallableSlot::REFLECTION,
         "__direct_callable__plasmaIonReflection"},
        {0, viennaray::gpu::CallableSlot::INIT,
         "__direct_callable__plasmaIonInit"},
        {1, viennaray::gpu::CallableSlot::COLLISION,
         "__direct_callable__plasmaNeutralCollision"},
        {1, viennaray::gpu::CallableSlot::REFLECTION,
         "__direct_callable__plasmaNeutralReflection"},
        {2, viennaray::gpu::CallableSlot::COLLISION,
         "__direct_callable__plasmaNeutralCollision"},
        {2, viennaray::gpu::CallableSlot::REFLECTION,
         "__direct_callable__plasmaNeutralReflection"}};
    this->setParticleCallableMap(pMap, cMap);

    this->setUseMaterialIds(true);
    precomputeSqrtEnergies();
    this->processData.alloc(sizeof(PlasmaEtchingParametersGPU));
    this->processData.upload(&deviceParams, 1);
    this->hasGPU = true;

    this->processMetaData = params.toProcessMetaData();
  }

  void setParameters(const PlasmaEtchingParameters<NumericType> &pParams) {
    params = pParams;
    deviceParams = PlasmaEtchingParametersGPU(pParams);
    precomputeSqrtEnergies();
    this->processData.upload(&deviceParams, 1);
  }

private:
  PlasmaEtchingParameters<NumericType> params;
  PlasmaEtchingParametersGPU deviceParams;

  void precomputeSqrtEnergies() {
    deviceParams.Substrate.Eth_ie = std::sqrt(deviceParams.Substrate.Eth_ie);
    deviceParams.Passivation.Eth_ie =
        std::sqrt(deviceParams.Passivation.Eth_ie);
    deviceParams.Substrate.Eth_sp = std::sqrt(deviceParams.Substrate.Eth_sp);
    deviceParams.Mask.Eth_sp = std::sqrt(deviceParams.Mask.Eth_sp);
    deviceParams.Polymer.Eth_sp = std::sqrt(deviceParams.Polymer.Eth_sp);
  }
};
} // namespace gpu
#endif

/// Model for etching Si in a CF4/Ar plasma.
template <typename NumericType, int D>
class CF4ArEtching : public ProcessModelCPU<NumericType, D> {
public:
  CF4ArEtching() {
    params = defaultParameters();
    initializeModel();
  }

  // All flux values are in units 1e15 / cm²
  CF4ArEtching(
      double ionFlux, double fluorineFlux, double cfxFlux,
      NumericType meanEnergy, NumericType sigmaEnergy,
      NumericType ionExponent = 300., NumericType cfxSputterYield = 3.,
      NumericType etchStopDepth = std::numeric_limits<NumericType>::lowest()) {
    params = defaultParameters();
    params.ionFlux = ionFlux;
    params.etchantFlux = fluorineFlux;
    params.passivationFlux = cfxFlux;
    params.Ions.meanEnergy = meanEnergy;
    params.Ions.sigmaEnergy = sigmaEnergy;
    params.Ions.exponent = ionExponent;
    params.Passivation.A_ie = cfxSputterYield;
    params.etchStopDepth = etchStopDepth;
    initializeModel();
  }

  CF4ArEtching(const PlasmaEtchingParameters<NumericType> &pParams)
      : params(pParams) {
    initializeModel();
  }

  void setParameters(const PlasmaEtchingParameters<NumericType> &pParams) {
    params = pParams;
    initializeModel();
  }

#ifdef VIENNACORE_COMPILE_GPU
  SmartPointer<ProcessModelBase<NumericType, D>> getGPUModel() override {
    auto model = SmartPointer<gpu::CF4ArEtching<NumericType, D>>::New(params);
    model->setProcessName(this->getProcessName().value());
    return model;
  }
#endif

  PlasmaEtchingParameters<NumericType> &getParameters() { return params; }

  static PlasmaEtchingParameters<NumericType> defaultParameters() {

    PlasmaEtchingParameters<NumericType> defParams;

    // fluxes in (1e15 /cm² /s)
    defParams.ionFlux = 12.;         // Ar+ ion flux
    defParams.etchantFlux = 1.8e3;   // F radical flux
    defParams.passivationFlux = 0.0; // lumped CFx flux (0 -> Step 1 baseline)

    // sticking probabilities
    defParams.beta_E.set(Material::Si, 0.7);   // F on Si
    defParams.beta_E.set(Material::Mask, 0.7); // F on mask

    defParams.beta_P.set(Material::Si, 1.);   // CFx on Si
    defParams.beta_P.set(Material::Mask, 1.); // CFx on mask

    defParams.etchStopDepth = std::numeric_limits<NumericType>::lowest();

    // Mask
    defParams.Mask.rho = 500.;   // 1e22 atoms/cm³
    defParams.Mask.Eth_sp = 20.; // eV
    defParams.Mask.A_sp = 0.0139;
    defParams.Mask.B_sp = 9.3;

    // Si substrate
    defParams.Substrate.rho = 5.02;   // 1e22 atoms/cm³
    defParams.Substrate.Eth_sp = 20.; // eV
    defParams.Substrate.Eth_ie = 15.; // eV
    defParams.Substrate.A_sp = 0.0337;
    defParams.Substrate.B_sp = 9.3;
    defParams.Substrate.A_ie = 7.;
    defParams.Substrate.B_ie = 0.8;
    defParams.Substrate.k_sigma = 3.0e2;     // chemical (F) etch coefficient
    defParams.Substrate.beta_sigma = 4.0e-2; // in (1e15 cm⁻²s⁻¹)

    // Passivation (Ar+ removal of lumped CFx residue)
    defParams.Passivation.Eth_ie = 10.; // eV
    defParams.Passivation.A_ie = 3;

    // Ions (Ar+)
    defParams.Ions.meanEnergy = 100.; // eV
    defParams.Ions.sigmaEnergy = 10.; // eV
    defParams.Ions.exponent = 500.;

    defParams.Ions.inflectAngle = 1.55334303;
    defParams.Ions.n_l = 10.;
    defParams.Ions.minAngle = 1.3962634;

    defParams.Ions.thetaRMin = constants::degToRad(70.);
    defParams.Ions.thetaRMax = constants::degToRad(90.);
    return defParams;
  }

private:
  void initializeModel() {
    // check if units have been set
    if (units::Length::getInstance().getUnit() == units::Length::UNDEFINED ||
        units::Time::getInstance().getUnit() == units::Time::UNDEFINED) {
      VIENNACORE_LOG_ERROR("Units have not been set.");
    }

    // particles
    this->particles.clear();
    if (params.ionFlux > 0) {
      auto ion =
          std::make_unique<impl::PlasmaEtchingIon<NumericType, D>>(params);
      this->insertNextParticleType(ion);
    }
    if (params.etchantFlux > 0) {
      auto fluorine =
          std::make_unique<impl::PlasmaEtchingNeutral<NumericType, D>>(
              "etchantFlux", params.beta_E, 2);
      this->insertNextParticleType(fluorine);
    }
    if (params.passivationFlux > 0) {
      auto cfx = std::make_unique<impl::PlasmaEtchingNeutral<NumericType, D>>(
          "passivationFlux", params.beta_P, 2);
      this->insertNextParticleType(cfx);
    }

    // surface model
    auto surfModel =
        SmartPointer<impl::PlasmaEtchingSurfaceModel<NumericType, D>>::New(
            params);
    this->setSurfaceModel(surfModel);

    // velocity field
    auto velField = SmartPointer<DefaultVelocityField<NumericType, D>>::New();
    this->setVelocityField(velField);

    this->setProcessName("CF4ArEtching");
    this->hasGPU = true;

    this->processMetaData = params.toProcessMetaData();
    // add units
    this->processMetaData["Units"] = std::vector<double>{
        static_cast<double>(units::Length::getInstance().getUnit()),
        static_cast<double>(units::Time::getInstance().getUnit())};
  }

  PlasmaEtchingParameters<NumericType> params;
};

PS_PRECOMPILE_PRECISION_DIMENSION(CF4ArEtching)

} // namespace viennaps
