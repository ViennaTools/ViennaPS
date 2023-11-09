#pragma once

#include <rayParticle.hpp>
#include <rayReflection.hpp>
#include <rayUtil.hpp>

template <typename NumericType, int D>
class Particle : public rayParticle<Particle<NumericType, D>, NumericType> {
public:
  Particle(const NumericType pSticking, const NumericType pPower)
      : stickingProbability(pSticking), sourcePower(pPower) {}
  void surfaceCollision(NumericType rayWeight,
                        const rayTriple<NumericType> &rayDir,
                        const rayTriple<NumericType> &geomNormal,
                        const unsigned int primID, const int materialId,
                        rayTracingData<NumericType> &localData,
                        const rayTracingData<NumericType> *globalData,
                        rayRNG &Rng) override final {
    // collect data for this hit
    localData.getVectorData(0)[primID] += rayWeight;
  }
  std::pair<NumericType, rayTriple<NumericType>>
  surfaceReflection(NumericType rayWeight, const rayTriple<NumericType> &rayDir,
                    const rayTriple<NumericType> &geomNormal,
                    const unsigned int primId, const int materialId,
                    const rayTracingData<NumericType> *globalData,
                    rayRNG &Rng) override final {
    double angle = rayInternal::PI / 6.;
    auto direction = rayReflectionConedCosine<NumericType, D>(
        rayDir, geomNormal, Rng, angle);
    // auto direction = rayReflectionSpecular<NumericType>(rayDir, geomNormal);

    return std::pair<NumericType, rayTriple<NumericType>>{stickingProbability,
                                                          direction};
  }
  void initNew(rayRNG &RNG) override final {}

  NumericType getSourceDistributionPower() const override final {
    return sourcePower;
  }
  std::vector<std::string> getLocalDataLabels() const override final {
    return {"particleRate"};
  }

private:
  const NumericType stickingProbability = 0.2;
  const NumericType sourcePower = 100.;
};
