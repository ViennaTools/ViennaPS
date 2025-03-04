#pragma once

#include <models/psMultiParticleProcess.hpp>
#include <psConstants.hpp>
#include <psMaterials.hpp>

#include <psgProcessModel.hpp>

#include <rayParticle.hpp>

namespace viennaps::gpu {

namespace impl {
struct IonParams {
  float thetaRMin = 0.f;
  float thetaRMax = 0.f;
  float minAngle = 0.f;
  float B_sp = 0.f;
  float meanEnergy = 0.f;
  float sigmaEnergy = 0.f;
  float thresholdEnergy = 0.f;
  float inflectAngle = 0.f;
  float n = 0.f;
};
} // namespace impl

using namespace viennacore;

template <typename NumericType, int D>
class MultiParticleProcess final : public ProcessModel<NumericType, D> {
public:
  MultiParticleProcess() {
    // surface model
    auto surfModel = SmartPointer<viennaps::impl::MultiParticleSurfaceModel<
        NumericType, D>>::New(fluxDataLabels_);

    // velocity field
    auto velField = SmartPointer<DefaultVelocityField<NumericType, D>>::New(2);

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("MultiParticleProcess");
    this->setPipelineFileName("MultiParticlePipeline");
  }

  void addNeutralParticle(NumericType stickingProbability,
                          const std::string &label = "neutralFlux") {
    std::string dataLabel = label + std::to_string(fluxDataLabels_.size());
    fluxDataLabels_.push_back(dataLabel);
    viennaray::gpu::Particle<NumericType> particle;
    particle.name = "Neutral";
    particle.sticking = stickingProbability;
    particle.dataLabels.push_back(dataLabel);
    setDirection(particle);
    particle.materialSticking[static_cast<int>(Material::Undefined)] =
        1.; // this will initialize all to default sticking

    this->insertNextParticleType(particle);
    this->setUseMaterialIds(true);
  }

  void
  addNeutralParticle(std::unordered_map<Material, NumericType> materialSticking,
                     NumericType defaultStickingProbability = 1.,
                     const std::string &label = "neutralFlux") {
    std::string dataLabel = label + std::to_string(fluxDataLabels_.size());
    fluxDataLabels_.push_back(dataLabel);

    viennaray::gpu::Particle<NumericType> particle;
    particle.name = "Neutral";
    particle.sticking = defaultStickingProbability;
    particle.dataLabels.push_back(dataLabel);
    setDirection(particle);
    for (auto &mat : materialSticking) {
      particle.materialSticking[static_cast<int>(mat.first)] = mat.second;
    }

    this->insertNextParticleType(particle);
    this->setUseMaterialIds(true);
  }

  void addIonParticle(NumericType sourcePower, NumericType thetaRMin = 0.,
                      NumericType thetaRMax = 90., NumericType minAngle = 80.,
                      NumericType B_sp = -1., NumericType meanEnergy = 0.,
                      NumericType sigmaEnergy = 0.,
                      NumericType thresholdEnergy = 0.,
                      NumericType inflectAngle = 0., NumericType n = 1,
                      const std::string &label = "ionFlux") {
    std::string dataLabel = label + std::to_string(fluxDataLabels_.size());
    fluxDataLabels_.push_back(dataLabel);

    viennaray::gpu::Particle<NumericType> particle;
    particle.name = "Ion";
    particle.dataLabels.push_back(dataLabel);
    setDirection(particle);
    particle.cosineExponent = sourcePower;

    impl::IonParams params;
    params.thetaRMin = constants::degToRad(thetaRMin);
    params.thetaRMax = constants::degToRad(thetaRMax);
    params.minAngle = constants::degToRad(minAngle);
    params.B_sp = B_sp;
    params.meanEnergy = meanEnergy;
    params.sigmaEnergy = sigmaEnergy;
    params.thresholdEnergy = thresholdEnergy;
    params.inflectAngle = constants::degToRad(inflectAngle);
    params.n = n;
    this->processData.allocUploadSingle(params);
    this->insertNextParticleType(particle);
  }

  void
  setRateFunction(std::function<NumericType(const std::vector<NumericType> &,
                                            const Material &)>
                      rateFunction) {
    auto surfModel = std::dynamic_pointer_cast<
        viennaps::impl::MultiParticleSurfaceModel<NumericType, D>>(
        this->getSurfaceModel());
    surfModel->rateFunction_ = rateFunction;
  }

private:
  std::vector<std::string> fluxDataLabels_;

  void setDirection(viennaray::gpu::Particle<NumericType> &particle) {
    auto direction = this->getPrimaryDirection();
    if (direction.has_value()) {
      particle.direction = direction.value();
    }
  }

  struct IonParams {
    NumericType sourcePower;
    NumericType thetaRMin;
    NumericType thetaRMax;
    NumericType minAngle;
    NumericType B_sp;
    NumericType meanEnergy;
    NumericType sigmaEnergy;
    NumericType thresholdEnergy;
    NumericType inflectAngle;
  };
};

} // namespace viennaps::gpu
