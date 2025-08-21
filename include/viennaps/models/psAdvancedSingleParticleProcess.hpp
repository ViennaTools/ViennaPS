#pragma once

#include "psSingleParticleProcess.hpp"

#include "../psMaterials.hpp"
#include "../psProcessModel.hpp"

#include <rayParticle.hpp>
#include <rayReflection.hpp>

namespace viennaps {

using namespace viennacore;

namespace impl {
template <typename NumericType, int D>
class SingleParticleExtended
    : public viennaray::Particle<SingleParticleExtended<NumericType, D>,
                                 NumericType> {
public:
  SingleParticleExtended(
      std::vector<std::pair<int, NumericType>> const &materialSticking,
      NumericType sticking, NumericType sourcePower, NumericType meanFreePath,
      bool fluxIncludeSticking)
      : materialSticking_(materialSticking), stickingProbability_(sticking),
        sourcePower_(sourcePower), meanFreePath_(meanFreePath),
        fluxIncludeSticking_(fluxIncludeSticking) {}

  void surfaceCollision(NumericType rayWeight, const Vec3D<NumericType> &,
                        const Vec3D<NumericType> &, const unsigned int primID,
                        const int material,
                        viennaray::TracingData<NumericType> &localData,
                        const viennaray::TracingData<NumericType> *,
                        RNG &) override final {
    NumericType stickingProbability = 1.0;
    if (fluxIncludeSticking_) {
      stickingProbability = stickingProbability_;
      for (const auto &pair : materialSticking_) {
        if (pair.first == material) {
          stickingProbability = pair.second;
          break;
        }
      }
    }
    localData.getVectorData(0)[primID] += rayWeight * stickingProbability;
  }
  std::pair<NumericType, Vec3D<NumericType>>
  surfaceReflection(NumericType, const Vec3D<NumericType> &,
                    const Vec3D<NumericType> &geomNormal, const unsigned int,
                    const int material,
                    const viennaray::TracingData<NumericType> *,
                    RNG &rngState) override final {
    auto stickingProbability = stickingProbability_;
    for (const auto &pair : materialSticking_) {
      if (pair.first == material) {
        stickingProbability = pair.second;
        break;
      }
    }
    auto direction =
        viennaray::ReflectionDiffuse<NumericType, D>(geomNormal, rngState);
    return std::pair<NumericType, Vec3D<NumericType>>{stickingProbability,
                                                      direction};
  }
  void initNew(RNG &) override final {}
  NumericType getSourceDistributionPower() const override final {
    return sourcePower_;
  }
  std::vector<std::string> getLocalDataLabels() const override final {
    return {"particleFlux"};
  }
  NumericType getMeanFreePath() const override final { return meanFreePath_; }

private:
  const std::vector<std::pair<int, NumericType>> materialSticking_;
  const NumericType stickingProbability_;
  const NumericType sourcePower_;
  const NumericType meanFreePath_;
  const bool fluxIncludeSticking_;
};
} // namespace impl

template <typename NumericType, int D>
class AdvancedSingleParticleProcess : public ProcessModel<NumericType, D> {
public:
  AdvancedSingleParticleProcess(
      std::unordered_map<Material, NumericType> materialRates,
      std::unordered_map<Material, NumericType> materialSticking,
      NumericType defaultRate = 1., NumericType defaultStickingProbability = 1.,
      NumericType sourceDistributionPower = 1., NumericType meanFreePath = -1.,
      bool fluxIncludeSticking = false) {

    std::vector<std::pair<int, NumericType>> materialStickingVec;
    for (const auto &pair : materialSticking) {
      materialStickingVec.emplace_back(static_cast<int>(pair.first),
                                       pair.second);
    }

    // particles
    auto particle =
        std::make_unique<impl::SingleParticleExtended<NumericType, D>>(
            materialStickingVec, defaultStickingProbability,
            sourceDistributionPower, meanFreePath, fluxIncludeSticking);

    // surface model
    auto surfModel =
        SmartPointer<impl::SingleParticleSurfaceModel<NumericType, D>>::New(
            defaultRate, materialRates);

    // velocity field
    auto velField = SmartPointer<DefaultVelocityField<NumericType, D>>::New(2);

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->insertNextParticleType(particle);
    this->setProcessName("AdvancedSingleParticleProcess");

    this->processMetaData["Default Rate"] =
        std::vector<NumericType>{defaultRate};
    this->processMetaData["Default Sticking Probability"] =
        std::vector<NumericType>{defaultStickingProbability};
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
    if (!materialSticking.empty()) {
      for (const auto &pair : materialSticking) {
        this->processMetaData[MaterialMap::getMaterialName(pair.first) +
                              " Sticking Probability"] =
            std::vector<NumericType>{pair.second};
      }
    }
  }
};

PS_PRECOMPILE_PRECISION_DIMENSION(AdvancedSingleParticleProcess)

} // namespace viennaps
