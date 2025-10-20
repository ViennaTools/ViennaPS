#pragma once

#include <models/psMultiParticleProcess.hpp>

#include <rayParticle.hpp>

namespace viennaps::gpu {

template <typename NumericType, int D>
class IonBeamEtching final : public ProcessModelGPU<NumericType, D> {
public:
  explicit IonBeamEtching(NumericType exponent) {
    viennaray::gpu::Particle<NumericType> particle{
        .name = "ion", .sticking = 0.f, .cosineExponent = exponent};
    particle.dataLabels.push_back("particleFlux");

    // surface model
    fluxDataLabels_.emplace_back("particleFlux");
    auto surfModel = SmartPointer<::viennaps::impl::MultiParticleSurfaceModel<
        NumericType, D>>::New(fluxDataLabels_);

    // velocity field
    auto velField =
        SmartPointer<::viennaps::DefaultVelocityField<NumericType, D>>::New();

    this->setPipelineFileName("IonBeamEtching");
    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->insertNextParticleType(particle);
    this->setProcessName("IonBeamEtching");
    this->processMetaData["exponent"] = {exponent};
  }

  void
  setRateFunction(std::function<NumericType(const std::vector<NumericType> &,
                                            const Material &)>
                      rateFunction) {
    auto surfModel = std::dynamic_pointer_cast<
        ::viennaps::impl::MultiParticleSurfaceModel<NumericType, D>>(
        this->getSurfaceModel());
    surfModel->rateFunction_ = rateFunction;
  }

private:
  std::vector<std::string> fluxDataLabels_;
};
} // namespace viennaps::gpu
