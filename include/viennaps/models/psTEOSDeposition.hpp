#pragma once

#include "../psProcessModel.hpp"

#include <rayParticle.hpp>

namespace viennaps {

using namespace viennacore;

namespace impl {
template <class NumericType>
class SingleTEOSSurfaceModel : public SurfaceModel<NumericType> {
  using SurfaceModel<NumericType>::coverages;
  const NumericType depositionRate;
  const NumericType reactionOrder;

public:
  SingleTEOSSurfaceModel(const NumericType passedRate,
                         const NumericType passedOrder)
      : depositionRate(passedRate), reactionOrder(passedOrder) {}

  SmartPointer<std::vector<NumericType>>
  calculateVelocities(SmartPointer<viennals::PointData<NumericType>> rates,
                      const std::vector<Vec3D<NumericType>> &coordinates,
                      const std::vector<NumericType> &materialIDs) override {
    // define the surface reaction here
    auto particleFlux = rates->getScalarData("particleFlux");
    std::vector<NumericType> velocity(particleFlux->size(), 0.);

    for (std::size_t i = 0; i < velocity.size(); i++) {
      // calculate surface velocity based on particle flux
      velocity[i] =
          depositionRate * std::pow(particleFlux->at(i), reactionOrder);
    }

    return SmartPointer<std::vector<NumericType>>::New(velocity);
  }

  void updateCoverages(SmartPointer<viennals::PointData<NumericType>> rates,
                       const std::vector<NumericType> &materialIDs) override {
    // update coverages based on fluxes
    auto particleFlux = rates->getScalarData("particleFlux");
    auto Coverage = coverages->getScalarData("Coverage");
    assert(Coverage->size() == particleFlux->size());

    for (std::size_t i = 0; i < Coverage->size(); i++) {
      Coverage->at(i) = std::min(particleFlux->at(i), NumericType(1));
    }
  }

  void initializeCoverages(unsigned numGeometryPoints) override {
    if (coverages == nullptr) {
      coverages = viennals::PointData<NumericType>::New();
    } else {
      coverages->clear();
    }
    std::vector<NumericType> cov(numGeometryPoints);
    coverages->insertNextScalarData(cov, "Coverage");
  }
};

template <class NumericType>
class MultiTEOSSurfaceModel : public SurfaceModel<NumericType> {
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

  SmartPointer<std::vector<NumericType>>
  calculateVelocities(SmartPointer<viennals::PointData<NumericType>> rates,
                      const std::vector<Vec3D<NumericType>> &coordinates,
                      const std::vector<NumericType> &materialIDs) override {
    // define the surface reaction here
    auto particleFluxP1 = rates->getScalarData("particleFluxP1");
    auto particleFluxP2 = rates->getScalarData("particleFluxP2");

    std::vector<NumericType> velocity(particleFluxP1->size(), 0.);

    for (std::size_t i = 0; i < velocity.size(); i++) {
      // calculate surface velocity based on particle fluxes
      velocity[i] =
          depositionRateP1 * std::pow(particleFluxP1->at(i), reactionOrderP1) +
          depositionRateP2 * std::pow(particleFluxP2->at(i), reactionOrderP2);
    }

    return SmartPointer<std::vector<NumericType>>::New(velocity);
  }
};

// Particle type (modify at you own risk)
template <class NumericType, int D>
class SingleTEOSParticle
    : public viennaray::Particle<SingleTEOSParticle<NumericType, D>,
                                 NumericType> {
public:
  SingleTEOSParticle(const NumericType pStickingProbability,
                     const NumericType pReactionOrder,
                     const std::string &pDataLabel = "particleFlux")
      : stickingProbability(pStickingProbability),
        reactionOrder(pReactionOrder), dataLabel(pDataLabel) {}
  std::pair<NumericType, Vec3D<NumericType>>
  surfaceReflection(NumericType, const Vec3D<NumericType> &,
                    const Vec3D<NumericType> &geomNormal,
                    const unsigned int primID, const int,
                    const viennaray::TracingData<NumericType> *globalData,
                    RNG &rngState) override final {
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
    auto direction =
        viennaray::ReflectionDiffuse<NumericType, D>(geomNormal, rngState);
    return std::pair<NumericType, Vec3D<NumericType>>{sticking, direction};
  }
  void surfaceCollision(NumericType rayWeight, const Vec3D<NumericType> &,
                        const Vec3D<NumericType> &, const unsigned int primID,
                        const int,
                        viennaray::TracingData<NumericType> &localData,
                        const viennaray::TracingData<NumericType> *,
                        RNG &) override final {
    localData.getVectorData(0)[primID] += rayWeight;
  }
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
class MultiTEOSParticle
    : public viennaray::Particle<MultiTEOSParticle<NumericType, D>,
                                 NumericType> {
public:
  MultiTEOSParticle(const NumericType pStickingProbability,
                    const std::string &pLabel)
      : stickingProbability(pStickingProbability), dataLabel(pLabel) {}
  std::pair<NumericType, Vec3D<NumericType>>
  surfaceReflection(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                    const Vec3D<NumericType> &geomNormal,
                    const unsigned int primID, const int materialId,
                    const viennaray::TracingData<NumericType> *globalData,
                    RNG &Rng) override final {
    auto direction =
        viennaray::ReflectionDiffuse<NumericType, D>(geomNormal, Rng);
    return std::pair<NumericType, Vec3D<NumericType>>{stickingProbability,
                                                      direction};
  }
  void surfaceCollision(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                        const Vec3D<NumericType> &geomNormal,
                        const unsigned int primID, const int materialId,
                        viennaray::TracingData<NumericType> &localData,
                        const viennaray::TracingData<NumericType> *globalData,
                        RNG &Rng) override final {
    localData.getVectorData(0)[primID] += rayWeight;
  }
  NumericType getSourceDistributionPower() const override final { return 1; }
  std::vector<std::string> getLocalDataLabels() const override final {
    return {dataLabel};
  }

private:
  const NumericType stickingProbability;
  const std::string dataLabel;
};
} // namespace impl

template <class NumericType, int D>
class TEOSDeposition : public ProcessModel<NumericType, D> {
public:
  TEOSDeposition(const NumericType pStickingP1, const NumericType pRateP1,
                 const NumericType pOrderP1, const NumericType pStickingP2 = 0.,
                 const NumericType pRateP2 = 0.,
                 const NumericType pOrderP2 = 0.) {
    // velocity field
    auto velField = SmartPointer<DefaultVelocityField<NumericType, D>>::New();
    this->setVelocityField(velField);

    if (pRateP2 == 0.) {
      // use single particle model

      // particle
      auto particle =
          std::make_unique<impl::SingleTEOSParticle<NumericType, D>>(
              pStickingP1, pOrderP1);

      // surface model
      auto surfModel =
          SmartPointer<impl::SingleTEOSSurfaceModel<NumericType>>::New(
              pRateP1, pOrderP1);

      this->setSurfaceModel(surfModel);
      this->insertNextParticleType(particle);
      this->setProcessName("SingleTEOSParticleTEOS");
    } else {
      // use multi (two) particle model

      // particles
      auto particle1 =
          std::make_unique<impl::MultiTEOSParticle<NumericType, D>>(
              pStickingP1, "particleFluxP1");
      auto particle2 =
          std::make_unique<impl::MultiTEOSParticle<NumericType, D>>(
              pStickingP2, "particleFluxP2");

      // surface model
      auto surfModel =
          SmartPointer<impl::MultiTEOSSurfaceModel<NumericType>>::New(
              pRateP1, pOrderP1, pRateP2, pOrderP2);

      this->setSurfaceModel(surfModel);
      this->insertNextParticleType(particle1);
      this->insertNextParticleType(particle2);
      this->setProcessName("MultiTEOSParticleTEOS");
    }
  }
};

} // namespace viennaps
