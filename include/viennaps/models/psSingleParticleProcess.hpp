#pragma once

#include "../psMaterials.hpp"
#include "../psProcessModel.hpp"

#include <rayParticle.hpp>
#include <rayReflection.hpp>

namespace SingleParticleImplementation {
template <typename NumericType, int D>
class SurfaceModel : public psSurfaceModel<NumericType> {
  const NumericType rateFactor_;
  const std::vector<psMaterial> maskMaterials_;

public:
  SurfaceModel(NumericType rate, const std::vector<psMaterial> &mask)
      : rateFactor_(rate), maskMaterials_(mask) {}

  psSmartPointer<std::vector<NumericType>> calculateVelocities(
      psSmartPointer<psPointData<NumericType>> rates,
      const std::vector<std::array<NumericType, 3>> &coordinates,
      const std::vector<NumericType> &materialIds) override {

    auto velocity =
        psSmartPointer<std::vector<NumericType>>::New(materialIds.size(), 0.);
    auto flux = rates->getScalarData("particleFlux");

    for (std::size_t i = 0; i < velocity->size(); i++) {
      if (!isMaskMaterial(materialIds[i])) {
        velocity->at(i) = flux->at(i) * rateFactor_;
      }
    }

    return velocity;
  }

private:
  bool isMaskMaterial(const NumericType &material) const {
    for (const auto &mat : maskMaterials_) {
      if (psMaterialMap::isMaterial(material, mat))
        return true;
    }
    return false;
  }
};

template <typename NumericType, int D>
class Particle : public rayParticle<Particle<NumericType, D>, NumericType> {
public:
  Particle(NumericType sticking, NumericType sourcePower)
      : stickingProbability_(sticking), sourcePower_(sourcePower) {}

  void surfaceCollision(NumericType rayWeight,
                        const rayTriple<NumericType> &rayDir,
                        const rayTriple<NumericType> &geomNormal,
                        const unsigned int primID, const int materialId,
                        rayTracingData<NumericType> &localData,
                        const rayTracingData<NumericType> *globalData,
                        rayRNG &Rng) override final {
    localData.getVectorData(0)[primID] += rayWeight;
  }
  std::pair<NumericType, rayTriple<NumericType>>
  surfaceReflection(NumericType rayWeight, const rayTriple<NumericType> &rayDir,
                    const rayTriple<NumericType> &geomNormal,
                    const unsigned int primID, const int materialId,
                    const rayTracingData<NumericType> *globalData,
                    rayRNG &Rng) override final {
    auto direction = rayReflectionDiffuse<NumericType, D>(geomNormal, Rng);
    return std::pair<NumericType, rayTriple<NumericType>>{stickingProbability_,
                                                          direction};
  }
  void initNew(rayRNG &RNG) override final {}
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
} // namespace SingleParticleImplementation

// Etching or deposition based on a single particle model with diffuse
// reflections.
template <typename NumericType, int D>
class psSingleParticleProcess : public psProcessModel<NumericType, D> {
public:
  psSingleParticleProcess(NumericType rate = 1.,
                          NumericType stickingProbability = 1.,
                          NumericType sourceDistributionPower = 1.,
                          psMaterial maskMaterial = psMaterial::None) {
    std::vector<psMaterial> maskMaterialVec = {maskMaterial};
    initialize(rate, stickingProbability, sourceDistributionPower,
               std::move(maskMaterialVec));
  }

  psSingleParticleProcess(NumericType rate, NumericType stickingProbability,
                          NumericType sourceDistributionPower,
                          std::vector<psMaterial> maskMaterial) {
    initialize(rate, stickingProbability, sourceDistributionPower,
               std::move(maskMaterial));
  }

private:
  void initialize(NumericType rate, NumericType stickingProbability,
                  NumericType sourceDistributionPower,
                  std::vector<psMaterial> &&maskMaterial) {
    // particles
    auto particle = std::make_unique<
        SingleParticleImplementation::Particle<NumericType, D>>(
        stickingProbability, sourceDistributionPower);

    // surface model
    auto surfModel = psSmartPointer<SingleParticleImplementation::SurfaceModel<
        NumericType, D>>::New(rate, maskMaterial);

    // velocity field
    auto velField = psSmartPointer<psDefaultVelocityField<NumericType>>::New(2);

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->insertNextParticleType(particle);
    this->setProcessName("SingleParticleProcess");
  }
};
