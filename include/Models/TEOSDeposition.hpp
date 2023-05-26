#pragma once

#include <psProcessModel.hpp>
#include <rayParticle.hpp>

template <class NumericType>
class SingleTEOSSurfaceModel : public psSurfaceModel<NumericType> {
  using psSurfaceModel<NumericType>::Coverages;
  const NumericType depositionRate;
  const NumericType reactionOrder;

public:
  SingleTEOSSurfaceModel(const NumericType passedRate,
                         const NumericType passedOrder)
      : depositionRate(passedRate), reactionOrder(passedOrder) {}

  psSmartPointer<std::vector<NumericType>> calculateVelocities(
      psSmartPointer<psPointData<NumericType>> Rates,
      const std::vector<std::array<NumericType, 3>> &coordinates,
      const std::vector<NumericType> &materialIDs) override {
    // define the surface reaction here
    auto particleFlux = Rates->getScalarData("particleFlux");
    std::vector<NumericType> velocity(particleFlux->size(), 0.);

    for (std::size_t i = 0; i < velocity.size(); i++) {
      // calculate surface velocity based on particle flux
      velocity[i] =
          depositionRate * std::pow(particleFlux->at(i), reactionOrder);
    }

    return psSmartPointer<std::vector<NumericType>>::New(velocity);
  }

  void
  updateCoverages(psSmartPointer<psPointData<NumericType>> Rates) override {
    // update coverages based on fluxes
    auto particleFlux = Rates->getScalarData("particleFlux");
    auto Coverage = Coverages->getScalarData("Coverage");
    assert(Coverage->size() == particleFlux->size());

    for (std::size_t i = 0; i < Coverage->size(); i++) {
      Coverage->at(i) = std::min(particleFlux->at(i), 1.);
    }
  }

  void initializeCoverages(unsigned numGeometryPoints) override {
    if (Coverages == nullptr) {
      Coverages = psSmartPointer<psPointData<NumericType>>::New();
    } else {
      Coverages->clear();
    }
    std::vector<NumericType> cov(numGeometryPoints);
    Coverages->insertNextScalarData(cov, "Coverage");
  }
};

template <class NumericType>
class MultiTEOSSurfaceModel : public psSurfaceModel<NumericType> {
  const NumericType depositionRateP1;
  const NumericType reactionOrderP1;
  const NumericType depositionRateP2;
  const NumericType reactionOrderP2;

public:
  MultiTEOSSurfaceModel(const NumericType passedRateP1,
                        const NumericType passedOrderP1,
                        const NumericType passedRateP2,
                        const NumericType passedOrderP2)
      : depositionRateP1(passedRateP1), reactionOrderP1(passedOrderP1),
        depositionRateP2(passedRateP2), reactionOrderP2(passedOrderP2) {}

  psSmartPointer<std::vector<NumericType>> calculateVelocities(
      psSmartPointer<psPointData<NumericType>> Rates,
      const std::vector<std::array<NumericType, 3>> &coordinates,
      const std::vector<NumericType> &materialIDs) override {
    // define the surface reaction here
    auto particleFluxP1 = Rates->getScalarData("particleFluxP1");
    auto particleFluxP2 = Rates->getScalarData("particleFluxP2");

    std::vector<NumericType> velocity(particleFluxP1->size(), 0.);

    for (std::size_t i = 0; i < velocity.size(); i++) {
      // calculate surface velocity based on particle fluxes
      velocity[i] =
          depositionRateP1 * std::pow(particleFluxP1->at(i), reactionOrderP1) +
          depositionRateP2 * std::pow(particleFluxP2->at(i), reactionOrderP2);
    }

    return psSmartPointer<std::vector<NumericType>>::New(velocity);
  }
};

// Particle type (modify at you own risk)
template <class NumericType, int D>
class TEOSSingleParticle
    : public rayParticle<TEOSSingleParticle<NumericType, D>, NumericType> {
public:
  TEOSSingleParticle(const NumericType pStickingProbability,
                     const NumericType pReactionOrder,
                     const std::string pDataLabel = "particleFlux")
      : stickingProbability(pStickingProbability),
        reactionOrder(pReactionOrder), dataLabel(pDataLabel) {}
  std::pair<NumericType, rayTriple<NumericType>>
  surfaceReflection(NumericType rayWeight, const rayTriple<NumericType> &rayDir,
                    const rayTriple<NumericType> &geomNormal,
                    const unsigned int primID, const int materialId,
                    const rayTracingData<NumericType> *globalData,
                    rayRNG &Rng) override final {
    const auto &cov = globalData->getVectorData(0)[primID];
    NumericType sticking;
    if (cov > 0.) {
      sticking = stickingProbability * std::pow(cov, reactionOrder - 1);
    } else {
      if (reactionOrder < 1.) {
        sticking = 1.;
      } else if (reactionOrder == 1.) {
        sticking = stickingProbability;
      } else {
        sticking = 0.;
      }
    }
    auto direction = rayReflectionDiffuse<NumericType, D>(geomNormal, Rng);
    return std::pair<NumericType, rayTriple<NumericType>>{sticking, direction};
  }
  void surfaceCollision(NumericType rayWeight,
                        const rayTriple<NumericType> &rayDir,
                        const rayTriple<NumericType> &geomNormal,
                        const unsigned int primID, const int materialId,
                        rayTracingData<NumericType> &localData,
                        const rayTracingData<NumericType> *globalData,
                        rayRNG &Rng) override final {
    localData.getVectorData(0)[primID] += rayWeight;
  }
  int getRequiredLocalDataSize() const override final { return 1; }
  NumericType getSourceDistributionPower() const override final { return 1; }
  std::vector<std::string> getLocalDataLabels() const override final {
    return {dataLabel};
  }

private:
  const NumericType stickingProbability;
  const NumericType reactionOrder;
  const std::string dataLabel = "particleFlux";
};

template <class NumericType, int D>
class TEOSMultiParticle
    : public rayParticle<TEOSMultiParticle<NumericType, D>, NumericType> {
public:
  TEOSMultiParticle(const NumericType pStickingProbability, std::string pLabel)
      : stickingProbability(pStickingProbability), dataLabel(pLabel) {}
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
  void surfaceCollision(NumericType rayWeight,
                        const rayTriple<NumericType> &rayDir,
                        const rayTriple<NumericType> &geomNormal,
                        const unsigned int primID, const int materialId,
                        rayTracingData<NumericType> &localData,
                        const rayTracingData<NumericType> *globalData,
                        rayRNG &Rng) override final {
    localData.getVectorData(0)[primID] += rayWeight;
  }
  int getRequiredLocalDataSize() const override final { return 1; }
  NumericType getSourceDistributionPower() const override final { return 1; }
  std::vector<std::string> getLocalDataLabels() const override final {
    return {dataLabel};
  }

private:
  const NumericType stickingProbability;
  const std::string dataLabel;
};

template <class NumericType, int D>
class TEOSDeposition : public psProcessModel<NumericType, D> {
public:
  TEOSDeposition(const NumericType pStickingP1, const NumericType pRateP1,
                 const NumericType pOrderP1, const NumericType pStickingP2 = 0.,
                 const NumericType pRateP2 = 0.,
                 const NumericType pOrderP2 = 0.) {
    // velocity field
    auto velField = psSmartPointer<psDefaultVelocityField<NumericType>>::New();
    this->setVelocityField(velField);

    if (pRateP2 == 0.) {
      // use single particle model

      // particle
      auto particle = std::make_unique<TEOSSingleParticle<NumericType, D>>(
          pStickingP1, pOrderP1);

      // surface model
      auto surfModel = psSmartPointer<SingleTEOSSurfaceModel<NumericType>>::New(
          pRateP1, pOrderP1);

      this->setSurfaceModel(surfModel);
      this->insertNextParticleType(particle);
      this->setProcessName("SingleParticleTEOS");
    } else {
      // use multi (two) particle model

      // particles
      auto particle1 = std::make_unique<TEOSMultiParticle<NumericType, D>>(
          pStickingP1, "particleFluxP1");
      auto particle2 = std::make_unique<TEOSMultiParticle<NumericType, D>>(
          pStickingP2, "particleFluxP2");

      // surface model
      auto surfModel = psSmartPointer<MultiTEOSSurfaceModel<NumericType>>::New(
          pRateP1, pOrderP1, pRateP2, pOrderP2);

      this->setSurfaceModel(surfModel);
      this->insertNextParticleType(particle1);
      this->insertNextParticleType(particle2);
      this->setProcessName("MultiParticleTEOS");
    }
  }
};