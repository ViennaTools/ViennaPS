#pragma once

#include <models/psSingleParticleProcess.hpp>
#include <psgProcessModel.hpp>
#include <raygParticle.hpp>

namespace viennaps::gpu {

template <typename NumericType, int D>
class SingleParticleProcess final : public ProcessModel<NumericType, D> {
public:
  explicit SingleParticleProcess(NumericType rate = 1.,
                                 NumericType stickingProbability = 1.,
                                 NumericType sourceExponent = 1.,
                                 Material maskMaterial = Material::Undefined) {
    std::unordered_map<Material, NumericType> maskMaterialMap = {
        {maskMaterial, 0.}};
    initialize(rate, stickingProbability, sourceExponent,
               std::move(maskMaterialMap));
  }

  SingleParticleProcess(NumericType rate, NumericType stickingProbability,
                        NumericType sourceExponent,
                        std::vector<Material> maskMaterial) {
    std::unordered_map<Material, NumericType> maskMaterialMap;
    for (auto &mat : maskMaterial) {
      maskMaterialMap[mat] = 0.;
    }
    initialize(rate, stickingProbability, sourceExponent,
               std::move(maskMaterialMap));
  }

  SingleParticleProcess(std::unordered_map<Material, NumericType> materialRates,
                        NumericType stickingProbability,
                        NumericType sourceExponent) {
    initialize(0., stickingProbability, sourceExponent,
               std::move(materialRates));
  }

private:
  void initialize(NumericType rate, NumericType stickingProbability,
                  NumericType sourceExponent,
                  std::unordered_map<Material, NumericType> &&maskMaterial) {
    // particles

    viennaray::gpu::Particle<NumericType> particle{
        .name = "SingleParticle",
        .sticking = stickingProbability,
        .cosineExponent = sourceExponent};
    particle.dataLabels.push_back("particleFlux");

    // surface model
    auto surfModel = SmartPointer<::viennaps::impl::SingleParticleSurfaceModel<
        NumericType, D>>::New(rate, maskMaterial);

    // velocity field
    auto velField =
        SmartPointer<::viennaps::DefaultVelocityField<NumericType, D>>::New(2);

    this->setPipelineFileName("SingleParticlePipeline");
    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->insertNextParticleType(particle);
    this->setProcessName("SingleParticleProcess");
  }
};
} // namespace viennaps::gpu
