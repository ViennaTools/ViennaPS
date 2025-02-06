#pragma once

#include <models/psMultiParticleProcess.hpp>

#include <pscuProcessModel.hpp>

namespace viennaps {

namespace gpu {

template <typename NumericType, int D>
class IonBeamEtching : public ProcessModel<NumericType, D> {
public:
  IonBeamEtching(NumericType exponent) {
    Particle<NumericType> particle{
        .name = "ion", .sticking = 0.f, .cosineExponent = exponent};
    particle.dataLabels.push_back("particleFlux");

    // surface model
    fluxDataLabels_.push_back("particleFlux");
    auto surfModel =
        SmartPointer<impl::MultiParticleSurfaceModel<NumericType, D>>::New(
            fluxDataLabels_);

    // velocity field
    auto velField =
        SmartPointer<::viennaps::DefaultVelocityField<NumericType, D>>::New(2);

    this->setPipelineFileName("IonBeamEtching");
    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->insertNextParticleType(particle);
    this->setProcessName("IonBeamEtching");
  }

  void
  setRateFunction(std::function<NumericType(const std::vector<NumericType> &,
                                            const Material &)>
                      rateFunction) {
    auto surfModel = std::dynamic_pointer_cast<
        impl::MultiParticleSurfaceModel<NumericType, D>>(
        this->getSurfaceModel());
    surfModel->rateFunction_ = rateFunction;
  }

private:
  std::vector<std::string> fluxDataLabels_;
};
} // namespace gpu
} // namespace viennaps