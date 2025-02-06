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

  SF6O2Etching(const SF6O2Parameters<NumericType> &pParams) : params(pParams) {
    initializeModel();
  }

  void setParameters(const SF6O2Parameters<NumericType> &pParams) {
    params = pParams;
  }

  auto &getParameters() { return params; }

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
    etchant.sticking = params.beta_F[int(Material::Si)];
    etchant.cosineExponent = 1.f;

    gpu::Particle<NumericType> oxygen;
    oxygen.name = "oxygen";
    oxygen.dataLabels.push_back("oxygenFlux");
    oxygen.sticking = params.beta_O[int(Material::Si)];
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
    this->insertNextParticleType(ion);
    this->insertNextParticleType(etchant);
    this->insertNextParticleType(oxygen);
    this->setPipelineFileName("SF6O2Pipeline");
  }

  SF6O2Parameters<NumericType> params;
};

} // namespace gpu
} // namespace viennaps