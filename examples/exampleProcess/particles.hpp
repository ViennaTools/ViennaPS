#pragma once

#include <rayParticle.hpp>
#include <rayReflection.hpp>
#include <rayUtil.hpp>

template <typename NumericType, int D>
class Particle
    : public viennaray::Particle<Particle<NumericType, D>, NumericType> {
public:
  Particle(const NumericType pSticking, const NumericType pPower)
      : stickingProbability(pSticking), sourcePower(pPower) {}
  void surfaceCollision(NumericType rayWeight,
                        const viennaray::Triple<NumericType> &rayDir,
                        const viennaray::Triple<NumericType> &geomNormal,
                        const unsigned int primID, const int materialId,
                        viennaray::TracingData<NumericType> &localData,
                        const viennaray::TracingData<NumericType> *globalData,
                        RNG &rngState) override final {
    // collect data for this hit
    localData.getVectorData(0)[primID] += rayWeight;
  }
  std::pair<NumericType, viennaray::Triple<NumericType>>
  surfaceReflection(NumericType rayWeight,
                    const viennaray::Triple<NumericType> &rayDir,
                    const viennaray::Triple<NumericType> &geomNormal,
                    const unsigned int primId, const int materialId,
                    const viennaray::TracingData<NumericType> *globalData,
                    RNG &rngState) override final {
    auto direction =
        viennaray::ReflectionDiffuse<NumericType, D>(geomNormal, rngState);
    return std::pair<NumericType, viennaray::Triple<NumericType>>{
        stickingProbability, direction};
  }
  void initNew(RNG &rngState) override final {}
  NumericType getSourceDistributionPower() const override final {
    return sourcePower;
  }
  std::vector<std::string> getLocalDataLabels() const override final {
    return std::vector<std::string>{"particleRate"};
  }

private:
  const NumericType stickingProbability = 0.2;
  const NumericType sourcePower = 1.;
};
