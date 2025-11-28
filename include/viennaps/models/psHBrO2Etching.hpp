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

#ifdef VIENNACORE_COMPILE_GPU
namespace gpu {
/// GPU Version of the HBr/O2 plasma etching model
template <typename NumericType, int D>
class HBrO2Etching final : public ProcessModelGPU<NumericType, D> {
public:
  explicit HBrO2Etching(const PlasmaEtchingParameters<NumericType> &pParams)
      : params(pParams), deviceParams(pParams.convertToFloat()) {
    initializeModel();
  }

  ~HBrO2Etching() override { this->processData.free(); }

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

    viennaray::gpu::Particle<NumericType> oxygen;
    oxygen.name = "Oxygen";
    oxygen.dataLabels.push_back("passivationFlux");
    oxygen.cosineExponent = 1.f;
    oxygen.materialSticking = params.beta_P;

    // surface model
    auto surfModel = SmartPointer<
        viennaps::impl::PlasmaEtchingSurfaceModel<NumericType, D>>::New(params);

    // velocity field
    auto velField = SmartPointer<DefaultVelocityField<NumericType, D>>::New();

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("HBrO2Etching");
    this->getParticleTypes().clear();
    this->hasGPU = true;

    this->insertNextParticleType(ion);
    this->insertNextParticleType(etchant);
    this->insertNextParticleType(oxygen);

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
    this->setCallableFileName("CallableWrapper");

    this->processData.alloc(sizeof(PlasmaEtchingParameters<float>));
    this->processData.upload(&deviceParams, 1);

    this->setUseMaterialIds(true);
    this->processMetaData = params.toProcessMetaData();
  }

  void setParameters(const PlasmaEtchingParameters<NumericType> &pParams) {
    params = pParams;
    deviceParams = pParams.convertToFloat();
    this->processData.upload(&deviceParams, 1);
  }

private:
  PlasmaEtchingParameters<NumericType> params;
  PlasmaEtchingParameters<float> deviceParams;
};
} // namespace gpu
#endif

/// Model for etching Si in a HBr/O2 plasma.
template <typename NumericType, int D>
class HBrO2Etching : public ProcessModelCPU<NumericType, D> {
public:
  HBrO2Etching() {
    params = defaultParameters();
    initializeModel();
  }

  // All flux values are in units 1e15 / cm²
  HBrO2Etching(
      double ionFlux, double etchantFlux, double oxygenFlux,
      NumericType meanEnergy, NumericType sigmaEnergy,
      NumericType ionExponent = 300., NumericType oxySputterYield = 2.,
      NumericType etchStopDepth = std::numeric_limits<NumericType>::lowest()) {
    params = defaultParameters();
    params.ionFlux = ionFlux;
    params.etchantFlux = etchantFlux;
    params.passivationFlux = oxygenFlux;
    params.Ions.meanEnergy = meanEnergy;
    params.Ions.sigmaEnergy = sigmaEnergy;
    params.Ions.exponent = ionExponent;
    params.Passivation.A_ie = oxySputterYield;
    params.etchStopDepth = etchStopDepth;
    initializeModel();
  }

  HBrO2Etching(const PlasmaEtchingParameters<NumericType> &pParams)
      : params(pParams) {
    initializeModel();
  }

  void setParameters(const PlasmaEtchingParameters<NumericType> &pParams) {
    params = pParams;
    initializeModel();
  }

#ifdef VIENNACORE_COMPILE_GPU
  SmartPointer<ProcessModelBase<NumericType, D>> getGPUModel() override {
    return SmartPointer<gpu::HBrO2Etching<NumericType, D>>::New(params);
  }
#endif

  PlasmaEtchingParameters<NumericType> &getParameters() { return params; }

  static PlasmaEtchingParameters<NumericType> defaultParameters() {

    PlasmaEtchingParameters<NumericType> defParams;

    // fluxes in (1e15 /cm² /s)
    defParams.ionFlux = 12.;
    defParams.etchantFlux = 1.8e3;
    defParams.passivationFlux = 1.0e2;

    // sticking probabilities
    defParams.beta_E = {{1, 0.1}, {0, 0.1}};
    defParams.beta_P = {{1, 1.}, {0, 1.}};

    defParams.etchStopDepth = std::numeric_limits<NumericType>::lowest();

    // Mask
    defParams.Mask.rho = 500.;   // 1e22 atoms/cm³
    defParams.Mask.Eth_sp = 20.; // eV
    defParams.Mask.A_sp = 0.0139;
    defParams.Mask.B_sp = 9.3;

    // Si
    defParams.Substrate.rho = 5.02;   // 1e22 atoms/cm³
    defParams.Substrate.Eth_sp = 30.; // eV
    defParams.Substrate.Eth_ie = 15.; // eV
    defParams.Substrate.A_sp = 0.5;
    defParams.Substrate.B_sp = 9.3;
    // defParams.Substrate.theta_g_sp = M_PI_2; // angle where yield is zero
    // [rad]
    defParams.Substrate.A_ie = 5.;
    defParams.Substrate.B_ie = 0.8;
    // defParams.Substrate.theta_g_ie =
    //     constants::degToRad(78);          // angle where yield is zero [rad]
    defParams.Substrate.k_sigma = 3.0e2;     // in (1e15 cm⁻²s⁻¹)
    defParams.Substrate.beta_sigma = 4.0e-2; // in (1e15 cm⁻²s⁻¹)

    // Passivation
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
    if (params.passivationFlux > 0) {
      auto oxygen =
          std::make_unique<impl::PlasmaEtchingNeutral<NumericType, D>>(
              "passivationFlux", params.beta_P, 2);
      this->insertNextParticleType(oxygen);
    }

    // surface model
    auto surfModel =
        SmartPointer<impl::PlasmaEtchingSurfaceModel<NumericType, D>>::New(
            params);
    this->setSurfaceModel(surfModel);

    // velocity field
    auto velField = SmartPointer<DefaultVelocityField<NumericType, D>>::New();
    this->setVelocityField(velField);

    this->setProcessName("HBrO2Etching");
    this->hasGPU = true;

    this->processMetaData = params.toProcessMetaData();
    // add units
    this->processMetaData["Units"] = std::vector<double>{
        static_cast<double>(units::Length::getInstance().getUnit()),
        static_cast<double>(units::Time::getInstance().getUnit())};
  }

  PlasmaEtchingParameters<NumericType> params;
};

PS_PRECOMPILE_PRECISION_DIMENSION(HBrO2Etching)

} // namespace viennaps
