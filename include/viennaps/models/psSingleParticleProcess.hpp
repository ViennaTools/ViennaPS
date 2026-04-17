#pragma once

#include "../materials/psMaterialValueMap.hpp"
#include "../process/psProcessModel.hpp"

#include <rayParticle.hpp>
#include <rayReflection.hpp>

namespace viennaps {

using namespace viennacore;

namespace impl {
template <typename NumericType, int D>
class SingleParticleSurfaceModel : public viennaps::SurfaceModel<NumericType> {
  MaterialValueMap<NumericType> &materialRates_;

public:
  SingleParticleSurfaceModel(MaterialValueMap<NumericType> &materialRates)
      : materialRates_(materialRates) {}

  SmartPointer<std::vector<NumericType>>
  calculateVelocities(SmartPointer<viennals::PointData<NumericType>> rates,
                      const std::vector<Vec3D<NumericType>> &coordinates,
                      const std::vector<NumericType> &materialIds) override {

    auto velocity =
        SmartPointer<std::vector<NumericType>>::New(materialIds.size(), 0.);
    auto flux = rates->getScalarData("particleFlux");
    assert(flux && velocity->size() == flux->size());

#pragma omp parallel for
    for (size_t i = 0; i < velocity->size(); i++) {
      velocity->at(i) =
          flux->at(i) *
          materialRates_.getRef(MaterialMap::mapToMaterial(materialIds[i]));
    }

    return velocity;
  }
};

template <typename NumericType, int D>
class SingleParticle
    : public viennaray::Particle<SingleParticle<NumericType, D>, NumericType> {
public:
  SingleParticle(MaterialValueMap<NumericType> &stickingProbabilities,
                 NumericType sourcePower)
      : stickingProbabilities_(stickingProbabilities), sourcePower_(sourcePower) {}

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
                    const int materialId,
                    const viennaray::TracingData<NumericType> *,
                    RNG &rngState) override final {
    auto sticking = stickingProbabilities_.get(Material::fromLegacyId(materialId));
    auto direction =
        viennaray::ReflectionDiffuse<NumericType, D>(geomNormal, rngState);
    return std::pair<NumericType, Vec3D<NumericType>>{sticking, direction};
  }
  void initNew(RNG &) override final {}
  NumericType getSourceDistributionPower() const override final {
    return sourcePower_;
  }
  std::vector<std::string> getLocalDataLabels() const override final {
    return {"particleFlux"};
  }

private:
  MaterialValueMap<NumericType> &stickingProbabilities_;
  const NumericType sourcePower_;
};
} // namespace impl

#ifdef VIENNACORE_COMPILE_GPU
namespace gpu {

template <typename NumericType, int D>
class SingleParticleProcess final : public ProcessModelGPU<NumericType, D> {
  NumericType sourceDistributionPower_ = 1.;
  MaterialValueMap<NumericType> materialRates_;
  MaterialValueMap<NumericType> stickingProbabilities_;

public:
  SingleParticleProcess(MaterialValueMap<NumericType> const &materialRates,
                        NumericType stickingProbability,
                        NumericType sourceExponent)
      : SingleParticleProcess(
            materialRates,
            MaterialValueMap<NumericType>::fromDefault(stickingProbability),
            sourceExponent) {}

  SingleParticleProcess(MaterialValueMap<NumericType> const &materialRates,
                        MaterialValueMap<NumericType> const &stickingProbabilities,
                        NumericType sourceExponent)
      : sourceDistributionPower_(sourceExponent), materialRates_(materialRates),
        stickingProbabilities_(stickingProbabilities) {
    viennaray::gpu::Particle<NumericType> particle{
        .name = "SingleParticle",
        .sticking = static_cast<NumericType>(stickingProbabilities_.getDefault()),
        .cosineExponent = sourceDistributionPower_};
    particle.dataLabels.push_back("particleFlux");
    for (auto entry : stickingProbabilities_) {
      particle.materialSticking[static_cast<int>(entry.material)] = entry.value;
    }
    // ensure materialSticking buffer is always allocated
    if (particle.materialSticking.empty()) {
      particle.materialSticking[static_cast<int>(Material::Undefined)] =
          static_cast<NumericType>(stickingProbabilities_.getDefault());
    }

    std::unordered_map<std::string, unsigned> pMap = {{"SingleParticle", 0}};
    std::vector<viennaray::gpu::CallableConfig> cMap = {
        {0, viennaray::gpu::CallableSlot::COLLISION,
         "__direct_callable__singleNeutralCollision"},
        {0, viennaray::gpu::CallableSlot::REFLECTION,
         "__direct_callable__singleNeutralReflectionPerMaterial"}};
    this->setParticleCallableMap(pMap, cMap);

    auto surfModel = SmartPointer<::viennaps::impl::SingleParticleSurfaceModel<
        NumericType, D>>::New(materialRates_);
    auto velField =
        SmartPointer<::viennaps::DefaultVelocityField<NumericType, D>>::New();

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->insertNextParticleType(particle);
    this->setProcessName("SingleParticleProcess");
    this->setUseMaterialIds(true);
    this->hasGPU = true;

    this->processMetaData["Default Sticking Probability"] =
        std::vector<double>{stickingProbabilities_.getDefault()};
    this->processMetaData["Source Exponent"] =
        std::vector<double>{sourceDistributionPower_};
    for (auto entry : stickingProbabilities_) {
      this->processMetaData[MaterialMap::toString(entry.material) +
                            " Sticking"] =
          std::vector<double>{static_cast<double>(entry.value)};
    }
    for (auto rate : materialRates_) {
      this->processMetaData[MaterialMap::toString(rate.material) + " Rate"] =
          std::vector<double>{rate.value};
    }
  }
};
} // namespace gpu
#endif

// Etching or deposition based on a single particle model with diffuse
// reflections.
template <typename NumericType, int D>
class SingleParticleProcess : public ProcessModelCPU<NumericType, D> {
  NumericType sourceDistributionPower_ = 1.;
  MaterialValueMap<NumericType> materialRates_;
  MaterialValueMap<NumericType> stickingProbabilities_;

public:
  SingleParticleProcess(NumericType rate = 1.,
                        NumericType stickingProbability = 1.,
                        NumericType sourceDistributionPower = 1.,
                        Material maskMaterial = Material::Undefined)
      : sourceDistributionPower_(sourceDistributionPower) {
    materialRates_.setDefault(rate);
    materialRates_.set(maskMaterial, 0.0);
    stickingProbabilities_.setDefault(stickingProbability);
    initialize();
  }

  SingleParticleProcess(NumericType rate, NumericType stickingProbability,
                        NumericType sourceDistributionPower,
                        std::vector<Material> maskMaterial)
      : sourceDistributionPower_(sourceDistributionPower) {
    materialRates_.setDefault(rate);
    for (auto &mat : maskMaterial) {
      materialRates_.set(mat, 0.0);
    }
    stickingProbabilities_.setDefault(stickingProbability);
    initialize();
  }

  SingleParticleProcess(std::unordered_map<Material, NumericType> materialRates,
                        NumericType stickingProbability,
                        NumericType sourceDistributionPower)
      : sourceDistributionPower_(sourceDistributionPower),
        materialRates_(materialRates) {
    materialRates_.setDefault(0.);
    stickingProbabilities_.setDefault(stickingProbability);
    initialize();
  }

  void setDefaultRate(NumericType rate) {
    materialRates_.setDefault(rate);
    this->processMetaData["Default Rate"] = std::vector<double>{rate};
  }

  void setMaterialRate(Material material, NumericType rate) {
    materialRates_.set(material, rate);
    this->processMetaData[MaterialMap::toString(material) + " Rate"] =
        std::vector<double>{rate};
  }

  void setDefaultStickingProbability(NumericType sticking) {
    stickingProbabilities_.setDefault(sticking);
    this->processMetaData["Default Sticking Probability"] =
        std::vector<double>{sticking};
  }

  void setMaterialStickingProbability(Material material, NumericType sticking) {
    stickingProbabilities_.set(material, sticking);
    this->processMetaData[MaterialMap::toString(material) + " Sticking"] =
        std::vector<double>{sticking};
  }

#ifdef VIENNACORE_COMPILE_GPU
  SmartPointer<ProcessModelBase<NumericType, D>> getGPUModel() final {
    auto model = SmartPointer<gpu::SingleParticleProcess<NumericType, D>>::New(
        materialRates_, stickingProbabilities_, sourceDistributionPower_);
    model->setProcessName(this->getProcessName().value_or("default"));
    return model;
  }
#endif

private:
  void initialize() {
    auto particle = std::make_unique<impl::SingleParticle<NumericType, D>>(
        stickingProbabilities_, sourceDistributionPower_);

    auto surfModel =
        SmartPointer<impl::SingleParticleSurfaceModel<NumericType, D>>::New(
            materialRates_);

    auto velField = SmartPointer<DefaultVelocityField<NumericType, D>>::New();

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->insertNextParticleType(particle);
    this->setProcessName("SingleParticleProcess");
    this->hasGPU = true;

    this->processMetaData["Default Rate"] =
        std::vector<double>{materialRates_.getDefault()};
    this->processMetaData["Default Sticking Probability"] =
        std::vector<double>{stickingProbabilities_.getDefault()};
    this->processMetaData["Source Exponent"] =
        std::vector<double>{sourceDistributionPower_};
    for (auto mrate : materialRates_) {
      this->processMetaData[MaterialMap::toString(mrate.material) + " Rate"] =
          std::vector<double>{mrate.value};
    }
  }
};

PS_PRECOMPILE_PRECISION_DIMENSION(SingleParticleProcess)

} // namespace viennaps
