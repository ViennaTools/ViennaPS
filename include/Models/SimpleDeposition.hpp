#pragma once

#include <rayParticle.hpp>
#include <rayReflection.hpp>
#include <rayUtil.hpp>

#include <psProcessModel.hpp>
#include <psSmartPointer.hpp>
#include <psSurfaceModel.hpp>
#include <psVelocityField.hpp>

template <typename NumericType, int D>
class SimpleDepositionSurfaceModel : public psSurfaceModel<NumericType> {
public:
  psSmartPointer<std::vector<NumericType>> calculateVelocities(
      psSmartPointer<psPointData<NumericType>> Rates,
      const std::vector<std::array<NumericType, 3>> &coordinates,
      const std::vector<NumericType> &materialIds) override {

    const auto depoRate = Rates->getScalarData("depoRate");
    return psSmartPointer<std::vector<NumericType>>::New(*depoRate);
  }
};

template <typename NumericType, int D>
class SimpleDepositionParticle
    : public rayParticle<SimpleDepositionParticle<NumericType, D>,
                         NumericType> {
public:
  SimpleDepositionParticle(const NumericType passedSticking,
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
  int getRequiredLocalDataSize() const override final { return 1; }
  NumericType getSourceDistributionPower() const override final {
    return sourcePower;
  }
  std::vector<std::string> getLocalDataLabels() const override final {
    return std::vector<std::string>{"depoRate"};
  }

private:
  const NumericType stickingProbability = 0.1;
  const NumericType sourcePower = 1.;
};

template <typename NumericType, int D>
class SimpleDeposition : public psProcessModel<NumericType, D> {
public:
  SimpleDeposition(const NumericType stickingProbability = 0.1,
                   const NumericType sourceDistributionPower = 1.) {
    // particles
    auto depoParticle =
        std::make_unique<SimpleDepositionParticle<NumericType, D>>(
            stickingProbability, sourceDistributionPower);

    // surface model
    auto surfModel =
        psSmartPointer<SimpleDepositionSurfaceModel<NumericType, D>>::New();

    // velocity field
    auto velField = psSmartPointer<psDefaultVelocityField<NumericType>>::New();

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("SimpleDeposition");
    this->insertNextParticleType(depoParticle);
  }
};