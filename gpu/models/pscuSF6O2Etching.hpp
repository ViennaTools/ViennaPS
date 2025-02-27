#pragma once

#include <curtParticle.hpp>
#include <pscuProcessModel.hpp>

#include <models/psSF6O2Etching.hpp>

namespace viennaps::gpu {

using namespace viennacore;

/// Model for etching Si in a SF6/O2 plasma. The model is based on the paper by
/// Belen et al., Vac. Sci. Technol. A 23, 99–113 (2005),
/// DOI: https://doi.org/10.1116/1.1830495
/// The resulting rate is in units of um / s.
template <typename NumericType, int D>
class SF6O2Etching final : public ProcessModel<NumericType, D> {
public:
  explicit SF6O2Etching(const SF6O2Parameters<NumericType> &pParams)
      : params(pParams) {
    setParameters(pParams);
    initializeModel();
  }

  ~SF6O2Etching() override { this->processData.free(); }

private:
  void initializeModel() {
    // particles
    gpu::Particle<NumericType> ion;
    ion.name = "ion";
    ion.dataLabels.push_back("ionSputterFlux");
    ion.dataLabels.push_back("ionEnhancedFlux");
    ion.dataLabels.push_back("ionEnhancedPassivationFlux");
    ion.sticking = 0.f;
    ion.cosineExponent = params.Ions.exponent;

    gpu::Particle<NumericType> etchant;
    etchant.name = "etchant";
    etchant.dataLabels.push_back("etchantFlux");
    etchant.sticking = params.beta_F[static_cast<int>(Material::Si)];
    etchant.cosineExponent = 1.f;

    gpu::Particle<NumericType> oxygen;
    oxygen.name = "oxygen";
    oxygen.dataLabels.push_back("oxygenFlux");
    oxygen.sticking = params.beta_O[static_cast<int>(Material::Si)];
    oxygen.cosineExponent = 1.f;

    // surface model
    auto surfModel =
        SmartPointer<viennaps::impl::SF6O2SurfaceModel<NumericType, D>>::New(
            params);

    // velocity field
    auto velField = SmartPointer<DefaultVelocityField<NumericType, D>>::New(2);

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("SF6O2Etching");
    this->getParticleTypes().clear();
    this->insertNextParticleType(ion);
    this->insertNextParticleType(etchant);
    this->insertNextParticleType(oxygen);
    this->setPipelineFileName("SF6O2Pipeline");

    this->processData.alloc(sizeof(SF6O2Parameters<float>));
    this->processData.upload(&deviceParams, 1);
  }

  void setParameters(const SF6O2Parameters<NumericType> &pParams) {
    if constexpr (std::is_same_v<NumericType, float>) {
      deviceParams = pParams;
    } else {
      deviceParams.ionFlux = static_cast<float>(pParams.ionFlux);
      deviceParams.etchantFlux = static_cast<float>(pParams.etchantFlux);
      deviceParams.oxygenFlux = static_cast<float>(pParams.oxygenFlux);

      for (auto &pair : pParams.beta_F) {
        deviceParams.beta_F[pair.first] = static_cast<float>(pair.second);
      }
      for (auto &pair : pParams.beta_O) {
        deviceParams.beta_O[pair.first] = static_cast<float>(pair.second);
      }

      deviceParams.etchStopDepth = static_cast<float>(pParams.etchStopDepth);
      deviceParams.fluxIncludeSticking = pParams.fluxIncludeSticking;

      deviceParams.Mask.A_sp = static_cast<float>(pParams.Mask.A_sp);
      deviceParams.Mask.B_sp = static_cast<float>(pParams.Mask.B_sp);
      deviceParams.Mask.Eth_sp = static_cast<float>(pParams.Mask.Eth_sp);
      deviceParams.Mask.rho = static_cast<float>(pParams.Mask.rho);

      deviceParams.Si.A_ie = static_cast<float>(pParams.Si.A_ie);
      deviceParams.Si.A_sp = static_cast<float>(pParams.Si.A_sp);
      deviceParams.Si.B_ie = static_cast<float>(pParams.Si.B_ie);
      deviceParams.Si.B_sp = static_cast<float>(pParams.Si.B_sp);
      deviceParams.Si.Eth_ie = static_cast<float>(pParams.Si.Eth_ie);
      deviceParams.Si.Eth_sp = static_cast<float>(pParams.Si.Eth_sp);
      deviceParams.Si.k_sigma = static_cast<float>(pParams.Si.k_sigma);
      deviceParams.Si.beta_sigma = static_cast<float>(pParams.Si.beta_sigma);
      deviceParams.Si.rho = static_cast<float>(pParams.Si.rho);

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

  SF6O2Parameters<NumericType> params;
  SF6O2Parameters<float> deviceParams;
};

} // namespace viennaps::gpu
