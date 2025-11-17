#pragma once

#include <rayParticle.hpp>
#include <rayReflection.hpp>
#include <rayUtil.hpp>

#include "../process/psProcessModel.hpp"
#include "../process/psSurfaceModel.hpp"
#include "../process/psVelocityField.hpp"
#include "../psConstants.hpp"
#include "../psUnits.hpp"

#include "psPlasmaEtching.hpp"
#include "psPlasmaEtchingParameters.hpp"

namespace viennaps {

using namespace viennacore;

#ifdef VIENNACORE_COMPILE_GPU
namespace gpu {
/// GPU Version of the SF6/C4F8 plasma etching model
template <typename NumericType, int D>
class SF6C4F8Etching final : public ProcessModelGPU<NumericType, D> {
public:
  explicit SF6C4F8Etching(const PlasmaEtchingParameters<NumericType> &pParams)
      : params(pParams), deviceParams(pParams.convertToFloat()) {
    initializeModel();
  }

  ~SF6C4F8Etching() override { this->processData.free(); }

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

    viennaray::gpu::Particle<NumericType> etchant;
    etchant.name = "Etchant";
    etchant.dataLabels.push_back("etchantFlux");
    etchant.cosineExponent = 1.f;
    etchant.materialSticking = params.beta_E;

    // No oxygen/passivation particle for SF6C4F8

    // surface model
    auto surfModel = SmartPointer<
        viennaps::impl::PlasmaEtchingSurfaceModel<NumericType, D>>::New(params);

    // velocity field
    auto velField = SmartPointer<DefaultVelocityField<NumericType, D>>::New();

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("SF6C4F8Etching");
    this->getParticleTypes().clear();

    this->insertNextParticleType(ion);
    this->insertNextParticleType(etchant);

    std::unordered_map<std::string, unsigned> pMap = {{"Ion", 0},
                                                      {"Etchant", 1}};
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
         "__direct_callable__plasmaNeutralReflectionNoPassivation"}};
    this->setParticleCallableMap(pMap, cMap);
    this->setCallableFileName("CallableWrapper");

    this->setUseMaterialIds(true);
    precomputeSqrtEnergies();
    this->processData.alloc(sizeof(PlasmaEtchingParameters<float>));
    this->processData.upload(&deviceParams, 1);
    this->hasGPU = true;

    this->processMetaData = params.toProcessMetaData();
  }

  void setParameters(const PlasmaEtchingParameters<NumericType> &pParams) {
    params = pParams;
    deviceParams = pParams.convertToFloat();
    precomputeSqrtEnergies();
    this->processData.upload(&deviceParams, 1);
  }

private:
  PlasmaEtchingParameters<NumericType> params;
  PlasmaEtchingParameters<float> deviceParams;

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

/// Model for etching Si in a SF6/C4F8 plasma. This model extends the SF6O2
/// etching model to include polymer etching. The polymer layer is deposited
/// by a separate process and etched by sputtering similarly to the mask
/// material. No passivation occurs in this model.
template <typename NumericType, int D>
class SF6C4F8Etching : public ProcessModelCPU<NumericType, D> {
public:
  SF6C4F8Etching() {
    params = defaultParameters();
    initializeModel();
  }

  // All flux values are in units 1e15 / cm²
  SF6C4F8Etching(
      double ionFlux, double etchantFlux, NumericType meanEnergy,
      NumericType sigmaEnergy, NumericType ionExponent = 300.,
      NumericType etchStopDepth = std::numeric_limits<NumericType>::lowest()) {
    params = defaultParameters();
    params.ionFlux = ionFlux;
    params.etchantFlux = etchantFlux;
    params.passivationFlux = 0.; // No passivation
    params.Ions.meanEnergy = meanEnergy;
    params.Ions.sigmaEnergy = sigmaEnergy;
    params.Ions.exponent = ionExponent;
    params.etchStopDepth = etchStopDepth;
    initializeModel();
  }

  SF6C4F8Etching(const PlasmaEtchingParameters<NumericType> &pParams)
      : params(pParams) {
    initializeModel();
  }

#ifdef VIENNACORE_COMPILE_GPU
  SmartPointer<ProcessModelBase<NumericType, D>> getGPUModel() override {
    return SmartPointer<gpu::SF6C4F8Etching<NumericType, D>>::New(params);
  }
#endif

  void setParameters(const PlasmaEtchingParameters<NumericType> &pParams) {
    params = pParams;
    initializeModel();
  }

  PlasmaEtchingParameters<NumericType> &getParameters() { return params; }

  static PlasmaEtchingParameters<NumericType> defaultParameters() {

    PlasmaEtchingParameters<NumericType> defParams;

    // fluxes in (1e15 /cm² /s)
    defParams.ionFlux = 12.;
    defParams.etchantFlux = 1.8e3;
    defParams.passivationFlux = 0.; // No passivation

    // sticking probabilities
    defParams.beta_E = {{1, 0.7}, {0, 0.7}};
    // No beta_P needed since passivationFlux = 0

    defParams.etchStopDepth = std::numeric_limits<NumericType>::lowest();

    // Mask
    defParams.Mask.rho = 500.;   // 1e22 atoms/cm³
    defParams.Mask.Eth_sp = 20.; // eV
    defParams.Mask.A_sp = 0.0139;
    defParams.Mask.B_sp = 9.3;

    // Polymer (C4F8) - deposited by separate process, etched by sputtering
    defParams.Polymer.rho = 5.0;    // 1e22 atoms/cm³
    defParams.Polymer.Eth_sp = 15.; // eV
    defParams.Polymer.A_sp = 0.02;
    defParams.Polymer.B_sp = 8.5;

    // Si
    defParams.Substrate.rho = 5.02;   // 1e22 atoms/cm³
    defParams.Substrate.Eth_sp = 20.; // eV
    defParams.Substrate.Eth_ie = 15.; // eV
    defParams.Substrate.A_sp = 0.0337;
    defParams.Substrate.B_sp = 9.3;
    defParams.Substrate.A_ie = 7.;
    defParams.Substrate.B_ie = 0.8;
    defParams.Substrate.k_sigma = 3.0e2;     // in (1e15 cm⁻²s⁻¹)
    defParams.Substrate.beta_sigma = 4.0e-2; // in (1e15 cm⁻²s⁻¹)

    // Passivation (unused in this model)
    defParams.Passivation.Eth_ie = 10.; // eV
    defParams.Passivation.A_ie = 3;

    // Ions
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
      Logger::getInstance().addError("Units have not been set.").print();
    }

    // particles
    this->particles.clear();
    if (params.ionFlux > 0) {
      auto ion =
          std::make_unique<impl::PlasmaEtchingIon<NumericType, D>>(params);
      this->insertNextParticleType(ion);
    }
    if (params.etchantFlux > 0) {
      auto etchant =
          std::make_unique<impl::PlasmaEtchingNeutral<NumericType, D>>(
              "etchantFlux", params.beta_E, 2);
      this->insertNextParticleType(etchant);
    }
    // No passivation particle since passivationFlux = 0

    // surface model
    auto surfModel =
        SmartPointer<impl::PlasmaEtchingSurfaceModel<NumericType, D>>::New(
            params);
    this->setSurfaceModel(surfModel);

    // velocity field
    auto velField = SmartPointer<DefaultVelocityField<NumericType, D>>::New();
    this->setVelocityField(velField);

    this->setProcessName("SF6C4F8Etching");
    this->hasGPU = true;

    processMetaData = params.toProcessMetaData();
    // add units
    processMetaData["Units"] = std::vector<double>{
        static_cast<double>(units::Length::getInstance().getUnit()),
        static_cast<double>(units::Time::getInstance().getUnit())};
  }

  PlasmaEtchingParameters<NumericType> params;
  using ProcessModelCPU<NumericType, D>::processMetaData;
};

PS_PRECOMPILE_PRECISION_DIMENSION(SF6C4F8Etching)

} // namespace viennaps
