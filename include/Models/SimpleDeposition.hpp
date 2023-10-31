#pragma once

#include <rayParticle.hpp>
#include <rayReflection.hpp>
#include <rayUtil.hpp>

#include <psProcessModel.hpp>
#include <psSmartPointer.hpp>
#include <psSurfaceModel.hpp>
#include <psVelocityField.hpp>

namespace SimpleDepositionImplementation {
template <typename NumericType, int D>
class SurfaceModel : public psSurfaceModel<NumericType> {
  const NumericType rate;

public:
  SurfaceModel(const NumericType pRate) : rate(pRate) {}

  psSmartPointer<std::vector<NumericType>> calculateVelocities(
      psSmartPointer<psPointData<NumericType>> Rates,
      const std::vector<std::array<NumericType, 3>> &coordinates,
      const std::vector<NumericType> &materialIds) override {

    auto depoRate = *Rates->getScalarData("depoRate");
    std::for_each(depoRate.begin(), depoRate.end(),
                  [this](NumericType &v) { v *= rate; });

    return psSmartPointer<std::vector<NumericType>>::New(std::move(depoRate));
  }
};

template <typename NumericType, int D>
class Particle : public rayParticle<Particle<NumericType, D>, NumericType> {
public:
  Particle(const NumericType passedSticking,
           const NumericType passedSourcePower)
      : stickingProbability(passedSticking), sourcePower(passedSourcePower) {}

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
    return std::pair<NumericType, rayTriple<NumericType>>{stickingProbability,
                                                          direction};
  }
  void initNew(rayRNG &RNG) override final {}
  NumericType getSourceDistributionPower() const override final {
    return sourcePower;
  }
  std::vector<std::string> getLocalDataLabels() const override final {
    return {"depoRate"};
  }

private:
  const NumericType stickingProbability = 0.1;
  const NumericType sourcePower = 1.;
};
} // namespace SimpleDepositionImplementation

template <typename NumericType, int D>
class SimpleDeposition : public psProcessModel<NumericType, D> {
public:
  SimpleDeposition(const NumericType rate = 1.,
                   const NumericType stickingProbability = 0.1,
                   const NumericType sourceDistributionPower = 1.) {
    // particles
    auto depoParticle = std::make_unique<
        SimpleDepositionImplementation::Particle<NumericType, D>>(
        stickingProbability, sourceDistributionPower);

    // surface model
    auto surfModel =
        psSmartPointer<SimpleDepositionImplementation::SurfaceModel<
            NumericType, D>>::New(rate);

    // velocity field
    auto velField = psSmartPointer<psDefaultVelocityField<NumericType>>::New();

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("SimpleDeposition");
    this->insertNextParticleType(depoParticle);
  }
};