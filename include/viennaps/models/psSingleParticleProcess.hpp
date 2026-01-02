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

#ifdef VIENNACORE_COMPILE_GPU
namespace gpu {

template <typename NumericType, int D>
class SingleParticleProcess final : public ProcessModelGPU<NumericType, D> {
public:
  SingleParticleProcess(std::unordered_map<Material, NumericType> materialRates,
                        NumericType rate, NumericType stickingProbability,
                        NumericType sourceExponent) {
    // particles
    viennaray::gpu::Particle<NumericType> particle{
        .name = "SingleParticle",
        .sticking = stickingProbability,
        .cosineExponent = sourceExponent};
    particle.dataLabels.push_back("particleFlux");

    std::unordered_map<std::string, unsigned> pMap = {{"SingleParticle", 0}};
    std::vector<viennaray::gpu::CallableConfig> cMap = {
        {0, viennaray::gpu::CallableSlot::COLLISION,
         "__direct_callable__singleNeutralCollision"},
        {0, viennaray::gpu::CallableSlot::REFLECTION,
         "__direct_callable__singleNeutralReflection"}};
    this->setParticleCallableMap(pMap, cMap);

    // surface model
    auto surfModel = SmartPointer<::viennaps::impl::SingleParticleSurfaceModel<
        NumericType, D>>::New(rate, materialRates);

    // velocity field
    auto velField =
        SmartPointer<::viennaps::DefaultVelocityField<NumericType, D>>::New();

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->insertNextParticleType(particle);
    this->setProcessName("SingleParticleProcess");
    this->hasGPU = true;

    this->processMetaData["Default Rate"] = std::vector<double>{rate};
    this->processMetaData["Sticking Probability"] =
        std::vector<double>{stickingProbability};
    this->processMetaData["Source Exponent"] =
        std::vector<double>{sourceExponent};
    if (!materialRates.empty()) {
      for (const auto &pair : materialRates) {
        if (pair.first == Material::Undefined)
          continue;
        this->processMetaData[MaterialMap::toString(pair.first) + " Rate"] =
            std::vector<double>{pair.second};
      }
    }
  }
};
} // namespace gpu
#endif

// Etching or deposition based on a single particle model with diffuse
// reflections.
template <typename NumericType, int D>
class SingleParticleProcess : public ProcessModelCPU<NumericType, D> {
  NumericType rate_ = 1.;
  NumericType stickingProbability_ = 1.;
  NumericType sourceDistributionPower_ = 1.;
  std::unordered_map<Material, NumericType> materialRates_;

public:
  SingleParticleProcess(NumericType rate = 1.,
                        NumericType stickingProbability = 1.,
                        NumericType sourceDistributionPower = 1.,
                        Material maskMaterial = Material::Undefined)
      : rate_(rate), stickingProbability_(stickingProbability),
        sourceDistributionPower_(sourceDistributionPower) {
    materialRates_[maskMaterial] = 0.;
    initialize();
  }

  SingleParticleProcess(NumericType rate, NumericType stickingProbability,
                        NumericType sourceDistributionPower,
                        std::vector<Material> maskMaterial)
      : rate_(rate), stickingProbability_(stickingProbability),
        sourceDistributionPower_(sourceDistributionPower) {
    for (auto &mat : maskMaterial) {
      materialRates_[mat] = 0.;
    }
    initialize();
  }

  SingleParticleProcess(std::unordered_map<Material, NumericType> materialRates,
                        NumericType stickingProbability,
                        NumericType sourceDistributionPower)
      : rate_(0.), stickingProbability_(stickingProbability),
        sourceDistributionPower_(sourceDistributionPower),
        materialRates_(std::move(materialRates)) {
    initialize();
  }

#ifdef VIENNACORE_COMPILE_GPU
  SmartPointer<ProcessModelBase<NumericType, D>> getGPUModel() final {
    auto model = SmartPointer<gpu::SingleParticleProcess<NumericType, D>>::New(
        materialRates_, rate_, stickingProbability_, sourceDistributionPower_);
    model->setProcessName(this->getProcessName().value_or("default"));
    return model;
  }
#endif

private:
  void initialize() {
    // particles
    auto particle = std::make_unique<impl::SingleParticle<NumericType, D>>(
        stickingProbability_, sourceDistributionPower_);

    // surface model
    auto surfModel =
        SmartPointer<impl::SingleParticleSurfaceModel<NumericType, D>>::New(
            rate_, materialRates_);

    // velocity field
    auto velField = SmartPointer<DefaultVelocityField<NumericType, D>>::New();

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->insertNextParticleType(particle);
    this->setProcessName("SingleParticleProcess");
    this->hasGPU = true;

    this->processMetaData["Default Rate"] = std::vector<double>{rate_};
    this->processMetaData["Sticking Probability"] =
        std::vector<double>{stickingProbability_};
    this->processMetaData["Source Exponent"] =
        std::vector<double>{sourceDistributionPower_};
    if (!materialRates_.empty()) {
      for (const auto &pair : materialRates_) {
        if (pair.first == Material::Undefined)
          continue; // skip undefined material

        this->processMetaData[MaterialMap::toString(pair.first) + " Rate"] =
            std::vector<double>{pair.second};
      }
    }
  }
};

PS_PRECOMPILE_PRECISION_DIMENSION(SingleParticleProcess)

} // namespace viennaps
