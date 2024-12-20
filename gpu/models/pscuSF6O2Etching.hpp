#pragma once

#include <psProcessModel.hpp>

#include <rayParticle.hpp>
#include <rayReflection.hpp>

#include <curtParticle.hpp>
#include <pscuProcessModel.hpp>

#include <models/psSF6O2Etching.hpp>

namespace viennaps {

namespace gpu {

using namespace viennacore;

/// Model for etching Si in a SF6/O2 plasma. The model is based on the paper by
/// Belen et al., Vac. Sci. Technol. A 23, 99–113 (2005),
/// DOI: https://doi.org/10.1116/1.1830495
/// The resulting rate is in units of um / s.
template <typename NumericType, int D>
class SF6O2Etching : public ProcessModel<NumericType, D> {
public:
  SF6O2Etching() { initializeModel(); }

  // All flux values are in units 1e16 / cm²
  SF6O2Etching(const double ionFlux, const double etchantFlux,
               const double oxygenFlux, const NumericType meanEnergy /* eV */,
               const NumericType sigmaEnergy /* eV */, // 5 parameters
               const NumericType ionExponent = 300.,
               const NumericType oxySputterYield = 2.,
               const NumericType etchStopDepth =
                   std::numeric_limits<NumericType>::lowest()) {
    params.ionFlux = ionFlux;
    params.etchantFlux = etchantFlux;
    params.oxygenFlux = oxygenFlux;
    params.Ions.meanEnergy = meanEnergy;
    params.Ions.sigmaEnergy = sigmaEnergy;
    params.Ions.exponent = ionExponent;
    params.Passivation.A_ie = oxySputterYield;
    params.etchStopDepth = etchStopDepth;
    initializeModel();
  }

  SF6O2Etching(const SF6O2Parameters<NumericType> &pParams) : params(pParams) {
    initializeModel();
  }

  void setParameters(const SF6O2Parameters<NumericType> &pParams) {
    params = pParams;
  }

  SF6O2Parameters<NumericType> &getParameters() { return params; }

private:
  void initializeModel() {
    // particles
    Particle<NumericType> ion;
    ion.name = "ion";
    ion.dataLabels.push_back("ionSputteringRate");
    ion.dataLabels.push_back("ionEnhancedRate");
    ion.dataLabels.push_back("oxygenSputteringRate");
    ion.sticking = 1.f;
    ion.cosineExponent = params.Ions.exponent;

    Particle<NumericType> etchant;
    etchant.name = "etchant";
    etchant.dataLabels.push_back("etchantRate");
    etchant.dataLabels.push_back("eSticking");
    etchant.sticking = params.beta_F;
    etchant.cosineExponent = 1.f;

    Particle<NumericType> oxygen;
    oxygen.name = "oxygen";
    oxygen.dataLabels.push_back("oxygenRate");
    oxygen.dataLabels.push_back("oSticking");
    oxygen.sticking = params.beta_O;
    oxygen.cosineExponent = 1.f;

    // surface model
    auto surfModel =
        SmartPointer<viennaps::impl::SF6O2SurfaceModel<NumericType, D>>::New(params);

    // velocity field
    auto velField = SmartPointer<DefaultVelocityField<NumericType>>::New(2);

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("SF6O2Etching");
    this->insertNextParticleType(ion);
    this->insertNextParticleType(etchant);
    this->insertNextParticleType(oxygen);
    this->setPipelineFileName("SF6O2Pipeline")
  }

  SF6O2Parameters<NumericType> params;
};

} // namespace gpu
} // namespace viennaps