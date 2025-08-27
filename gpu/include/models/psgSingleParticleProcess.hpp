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

  SingleParticleProcess(std::unordered_map<Material, NumericType> materialRates,
                        NumericType rate, NumericType stickingProbability,
                        NumericType sourceExponent) {
    initialize(rate, stickingProbability, sourceExponent,
               std::move(materialRates));
  }

private:
  void initialize(NumericType rate, NumericType stickingProbability,
                  NumericType sourceExponent,
                  std::unordered_map<Material, NumericType> &&materialRates) {
    // particles
    viennaray::gpu::Particle<NumericType> particle{
        .name = "SingleParticle",
        .sticking = stickingProbability,
        .cosineExponent = sourceExponent};
    particle.dataLabels.push_back("particleFlux");

    // surface model
    auto surfModel = SmartPointer<::viennaps::impl::SingleParticleSurfaceModel<
        NumericType, D>>::New(rate, materialRates);

    // velocity field
    auto velField =
        SmartPointer<::viennaps::DefaultVelocityField<NumericType, D>>::New(2);

    this->setPipelineFileName("SingleParticlePipeline");
    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->insertNextParticleType(particle);
    this->setProcessName("SingleParticleProcess");

    this->processMetaData["Default Rate"] = std::vector<NumericType>{rate};
    this->processMetaData["Sticking Probability"] =
        std::vector<NumericType>{stickingProbability};
    this->processMetaData["Source Exponent"] =
        std::vector<NumericType>{sourceExponent};
    if (!materialRates.empty()) {
      for (const auto &pair : materialRates) {
        if (pair.first == Material::Undefined)
          continue; // skip undefined material

        this->processMetaData[MaterialMap::getMaterialName(pair.first) +
                              " Rate"] = std::vector<NumericType>{pair.second};
      }
    }
  }
};
} // namespace viennaps::gpu
