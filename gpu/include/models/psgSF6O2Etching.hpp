#pragma once

#include <models/psSF6O2Etching.hpp>
#include <process/psProcessModel.hpp>
#include <raygParticle.hpp>

namespace viennaps::gpu {

using namespace viennacore;

/// Model for etching Si in a SF6/O2 plasma. The model is based on the paper by
/// Belen et al., Vac. Sci. Technol. A 23, 99–113 (2005),
/// DOI: https://doi.org/10.1116/1.1830495
/// The resulting rate is in units of um / s.
template <typename NumericType, int D>
class SF6O2Etching final : public ProcessModel<NumericType, D> {
public:
  explicit SF6O2Etching(const PlasmaEtchingParameters<NumericType> &pParams)
      : params(pParams) {
    setParameters(pParams);
    initializeModel();
  }

  ~SF6O2Etching() override { this->processData.free(); }

private:
  void initializeModel() {
    // particles
    viennaray::gpu::Particle<NumericType> ion;
    ion.name = "ion";
    ion.dataLabels.push_back("ionSputterFlux");
    ion.dataLabels.push_back("ionEnhancedFlux");
    ion.dataLabels.push_back("ionEnhancedPassivationFlux");
    ion.sticking = 0.f;
    ion.cosineExponent = params.Ions.exponent;

    viennaray::gpu::Particle<NumericType> etchant;
    etchant.name = "neutral";
    etchant.dataLabels.push_back("etchantFlux");
    etchant.cosineExponent = 1.f;
    etchant.materialSticking = params.beta_E;

    viennaray::gpu::Particle<NumericType> oxygen;
    oxygen.name = "neutral";
    oxygen.dataLabels.push_back("passivationFlux");
    oxygen.cosineExponent = 1.f;
    oxygen.materialSticking = params.beta_P;

    // surface model
    auto surfModel = SmartPointer<
        viennaps::impl::PlasmaEtchingSurfaceModel<NumericType, D>>::New(params);

    // velocity field
    auto velField = SmartPointer<DefaultVelocityField<NumericType, D>>::New(2);

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("SF6O2Etching");
    this->getParticleTypes().clear();

    this->insertNextParticleType(ion);
    this->insertNextParticleType(etchant);
    this->insertNextParticleType(oxygen);
    this->setPipelineFileName("PlasmaEtchingPipeline");

    this->processData.alloc(sizeof(PlasmaEtchingParameters<float>));
    this->processData.upload(&deviceParams, 1);

    this->setUseMaterialIds(true);
    this->processMetaData = params.toProcessMetaData();
  }

  void setParameters(const PlasmaEtchingParameters<NumericType> &pParams) {
    if constexpr (std::is_same_v<NumericType, float>) {
      deviceParams = pParams;
    } else {
      deviceParams.ionFlux = static_cast<float>(pParams.ionFlux);
      deviceParams.etchantFlux = static_cast<float>(pParams.etchantFlux);
      deviceParams.passivationFlux =
          static_cast<float>(pParams.passivationFlux);

      for (auto &pair : pParams.beta_E) {
        deviceParams.beta_E[pair.first] = static_cast<float>(pair.second);
      }
      for (auto &pair : pParams.beta_P) {
        deviceParams.beta_P[pair.first] = static_cast<float>(pair.second);
      }

      deviceParams.etchStopDepth = static_cast<float>(pParams.etchStopDepth);

      deviceParams.Mask.A_sp = static_cast<float>(pParams.Mask.A_sp);
      deviceParams.Mask.B_sp = static_cast<float>(pParams.Mask.B_sp);
      deviceParams.Mask.Eth_sp = static_cast<float>(pParams.Mask.Eth_sp);
      deviceParams.Mask.rho = static_cast<float>(pParams.Mask.rho);

      deviceParams.Substrate.A_ie = static_cast<float>(pParams.Substrate.A_ie);
      deviceParams.Substrate.A_sp = static_cast<float>(pParams.Substrate.A_sp);
      deviceParams.Substrate.B_ie = static_cast<float>(pParams.Substrate.B_ie);
      deviceParams.Substrate.B_sp = static_cast<float>(pParams.Substrate.B_sp);
      deviceParams.Substrate.Eth_ie =
          static_cast<float>(pParams.Substrate.Eth_ie);
      deviceParams.Substrate.Eth_sp =
          static_cast<float>(pParams.Substrate.Eth_sp);
      deviceParams.Substrate.k_sigma =
          static_cast<float>(pParams.Substrate.k_sigma);
      deviceParams.Substrate.beta_sigma =
          static_cast<float>(pParams.Substrate.beta_sigma);
      deviceParams.Substrate.rho = static_cast<float>(pParams.Substrate.rho);

      deviceParams.Passivation.A_ie =
          static_cast<float>(pParams.Passivation.A_ie);
      deviceParams.Passivation.Eth_ie =
          static_cast<float>(pParams.Passivation.Eth_ie);

      deviceParams.Ions.exponent = static_cast<float>(pParams.Ions.exponent);
      deviceParams.Ions.meanEnergy =
          static_cast<float>(pParams.Ions.meanEnergy);
      deviceParams.Ions.sigmaEnergy =
          static_cast<float>(pParams.Ions.sigmaEnergy);

      deviceParams.Ions.inflectAngle =
          static_cast<float>(pParams.Ions.inflectAngle);
      deviceParams.Ions.n_l = static_cast<float>(pParams.Ions.n_l);
      deviceParams.Ions.minAngle = static_cast<float>(pParams.Ions.minAngle);

      deviceParams.Ions.thetaRMin = static_cast<float>(pParams.Ions.thetaRMin);
      deviceParams.Ions.thetaRMax = static_cast<float>(pParams.Ions.thetaRMax);
    }
  }

  PlasmaEtchingParameters<NumericType> params;
  PlasmaEtchingParameters<float> deviceParams;
};

} // namespace viennaps::gpu
