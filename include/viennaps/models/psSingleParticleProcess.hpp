#pragma once

#include "../process/psProcessModel.hpp"
#include "../psMaterials.hpp"

#include <rayParticle.hpp>
#include <rayReflection.hpp>

namespace viennaps {

using namespace viennacore;

namespace impl {
template <typename NumericType, int D>
class SingleParticleSurfaceModel : public viennaps::SurfaceModel<NumericType> {
  const NumericType rate_;
  const std::unordered_map<Material, NumericType> materialRates_;

public:
  SingleParticleSurfaceModel(
      NumericType rate, const std::unordered_map<Material, NumericType> &mask)
      : rate_(rate), materialRates_(mask) {}

  SmartPointer<std::vector<NumericType>>
  calculateVelocities(SmartPointer<viennals::PointData<NumericType>> rates,
                      const std::vector<Vec3D<NumericType>> &coordinates,
                      const std::vector<NumericType> &materialIds) override {

    auto velocity =
        SmartPointer<std::vector<NumericType>>::New(materialIds.size(), 0.);
    auto flux = rates->getScalarData("particleFlux");

#pragma omp parallel for
    for (size_t i = 0; i < velocity->size(); i++) {
      if (auto matRate =
              materialRates_.find(MaterialMap::mapToMaterial(materialIds[i]));
          matRate == materialRates_.end()) {
        velocity->at(i) = flux->at(i) * rate_;
      } else {
        velocity->at(i) = flux->at(i) * matRate->second;
      }
    }

    return velocity;
  }
};

template <typename NumericType, int D>
class SingleParticle
    : public viennaray::Particle<SingleParticle<NumericType, D>, NumericType> {
public:
  SingleParticle(NumericType sticking, NumericType sourcePower)
      : stickingProbability_(sticking), sourcePower_(sourcePower) {}

  void surfaceCollision(NumericType rayWeight, const Vec3D<NumericType> &,
                        const Vec3D<NumericType> &, const unsigned int primID,
                        const int,
                        viennaray::TracingData<NumericType> &localData,
                        const viennaray::TracingData<NumericType> *,
                        RNG &) override final {
    localData.getVectorData(0)[primID] += rayWeight;
  }
  std::pair<NumericType, Vec3D<NumericType>>
  surfaceReflection(NumericType, const Vec3D<NumericType> &,
                    const Vec3D<NumericType> &geomNormal, const unsigned int,
                    const int, const viennaray::TracingData<NumericType> *,
                    RNG &rngState) override final {
    auto direction =
        viennaray::ReflectionDiffuse<NumericType, D>(geomNormal, rngState);
    return std::pair<NumericType, Vec3D<NumericType>>{stickingProbability_,
                                                      direction};
  }
  void initNew(RNG &) override final {}
  NumericType getSourceDistributionPower() const override final {
    return sourcePower_;
  }
  std::vector<std::string> getLocalDataLabels() const override final {
    return {"particleFlux"};
  }

private:
  const NumericType stickingProbability_;
  const NumericType sourcePower_;
};
} // namespace impl

// Etching or deposition based on a single particle model with diffuse
// reflections.
template <typename NumericType, int D>
class SingleParticleProcess : public ProcessModel<NumericType, D> {
public:
  SingleParticleProcess(NumericType rate = 1.,
                        NumericType stickingProbability = 1.,
                        NumericType sourceDistributionPower = 1.,
                        Material maskMaterial = Material::Undefined) {
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
                  std::unordered_map<Material, NumericType> &&materialRates) {
    // particles
    auto particle = std::make_unique<impl::SingleParticle<NumericType, D>>(
        stickingProbability, sourceDistributionPower);

    // surface model
    auto surfModel =
        SmartPointer<impl::SingleParticleSurfaceModel<NumericType, D>>::New(
            rate, materialRates);

    // velocity field
    auto velField = SmartPointer<DefaultVelocityField<NumericType, D>>::New(2);

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->insertNextParticleType(particle);
    this->setProcessName("SingleParticleProcess");

    this->processMetaData["Default Rate"] = std::vector<NumericType>{rate};
    this->processMetaData["Sticking Probability"] =
        std::vector<NumericType>{stickingProbability};
    this->processMetaData["Source Exponent"] =
        std::vector<NumericType>{sourceDistributionPower};
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

PS_PRECOMPILE_PRECISION_DIMENSION(SingleParticleProcess)

} // namespace viennaps
