#pragma once

#include <models/psSingleParticleProcess.hpp>

#include <pscuProcessModel.hpp>

namespace viennaps {

namespace gpu {

template <typename NumericType, int D>
class SingleParticleProcess : public ProcessModel<NumericType, D> {
public:
  SingleParticleProcess(NumericType rate = 1.,
                        NumericType stickingProbability = 1.,
                        NumericType sourceDistributionPower = 1.,
                        Material maskMaterial = Material::None) {
    std::unordered_map<Material, NumericType> maskMaterialMap = {
        {maskMaterial, 0.}};
    initialize(rate, stickingProbability, sourceDistributionPower,
               std::move(maskMaterialMap));
  }

  SingleParticleProcess(NumericType rate, NumericType stickingProbability,
                        NumericType sourceDistributionPower,
                        std::vector<Material> maskMaterial) {
    std::unordered_map<Material, NumericType> maskMaterialMap;
    for (auto &mat : maskMaterial) {
      maskMaterialMap[mat] = 0.;
    }
    initialize(rate, stickingProbability, sourceDistributionPower,
               std::move(maskMaterialMap));
  }

  SingleParticleProcess(std::unordered_map<Material, NumericType> materialRates,
                        NumericType stickingProbability,
                        NumericType sourceDistributionPower) {
    initialize(0., stickingProbability, sourceDistributionPower,
               std::move(materialRates));
  }

private:
  void initialize(NumericType rate, NumericType stickingProbability,
                  NumericType sourceDistributionPower,
                  std::unordered_map<Material, NumericType> &&maskMaterial) {
    // particles

    Particle<NumericType> particle{.name = "SingleParticle",
                                   .sticking = stickingProbability,
                                   .cosineExponent = sourceDistributionPower};
    particle.dataLabels.push_back("particleFlux");

    // surface model
    auto surfModel = SmartPointer<::viennaps::impl::SingleParticleSurfaceModel<
        NumericType, D>>::New(rate, maskMaterial);

    // velocity field
    auto velField =
        SmartPointer<::viennaps::DefaultVelocityField<NumericType>>::New(2);

    this->setPipelineFileName("SingleParticlePipeline");
    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->insertNextParticleType(particle);
    this->setProcessName("SingleParticleProcess");
  }
};
} // namespace gpu
} // namespace viennaps