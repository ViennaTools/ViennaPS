#pragma once

#include "../psProcessModel.hpp"

#include <rayParticle.hpp>
#include <rayReflection.hpp>
#include <rayUtil.hpp>

namespace viennaps {

namespace impl {
template <class NumericType>
class PECVDSurfaceModel : public SurfaceModel<NumericType> {
  const NumericType radicalRate;
  const NumericType radicalReactionOrder;
  const NumericType ionRate;
  const NumericType ionReactionOrder;

public:
  PECVDSurfaceModel(const NumericType passedRadicalRate,
                    const NumericType passedRadicalReactionOrder,
                    const NumericType passedIonRate,
                    const NumericType passedIonReactionOrder)
      : radicalRate(passedRadicalRate),
        radicalReactionOrder(passedRadicalReactionOrder),
        ionRate(passedIonRate), ionReactionOrder(passedIonReactionOrder) {}

  SmartPointer<std::vector<NumericType>> calculateVelocities(
      SmartPointer<viennals::PointData<NumericType>> rates,
      const std::vector<std::array<NumericType, 3>> &coordinates,
      const std::vector<NumericType> &materialIDs) override {
    // define the surface reaction here
    auto particleFluxRadical = rates->getScalarData("radicalFlux");
    auto particleFluxIon = rates->getScalarData("ionFlux");

    std::vector<NumericType> velocity(particleFluxRadical->size(), 0.);

    for (std::size_t i = 0; i < velocity.size(); i++) {
      // calculate surface velocity based on particle fluxes
      velocity[i] =
          radicalRate *
              std::pow(particleFluxRadical->at(i), radicalReactionOrder) +
          ionRate * std::pow(particleFluxIon->at(i), ionReactionOrder);
    }

    return SmartPointer<std::vector<NumericType>>::New(velocity);
  }
};

// Particle type (modify at you own risk)
template <class NumericType, int D>
class Radical
    : public viennaray::Particle<Radical<NumericType, D>, NumericType> {
public:
  Radical(const NumericType pStickingProbability,
          const std::string pDataLabel = "radicalFlux")
      : stickingProbability(pStickingProbability), dataLabel(pDataLabel) {}
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
  const std::string dataLabel = "radicalFlux";
};

template <typename NumericType, int D>
class Ion : public viennaray::Particle<Ion<NumericType, D>, NumericType> {
public:
  Ion(const NumericType pStickingProbability, const NumericType pExponent,
      const NumericType pMinAngle, const std::string pDataLabel = "ionFlux")
      : stickingProbability(pStickingProbability), exponent(pExponent),
        minAngle(pMinAngle), dataLabel(pDataLabel) {}

  std::pair<NumericType, Vec3D<NumericType>>
  surfaceReflection(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                    const Vec3D<NumericType> &geomNormal,
                    const unsigned int primId, const int materialId,
                    const viennaray::TracingData<NumericType> *globalData,
                    RNG &Rng) override final {
    auto cosTheta = -rayInternal::DotProduct(rayDir, geomNormal);

    assert(cosTheta >= 0 && "Hit backside of disc");
    assert(cosTheta <= 1 + 1e-6 && "Error in calculating cos theta");

    NumericType incAngle =
        std::acos(std::max(std::min(cosTheta, static_cast<NumericType>(1.)),
                           static_cast<NumericType>(0.)));

    auto direction = viennaray::ReflectionConedCosine<NumericType, D>(
        rayDir, geomNormal, Rng, std::max(incAngle, minAngle));
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

  NumericType getSourceDistributionPower() const override final {
    return exponent;
  }
  std::vector<std::string> getLocalDataLabels() const override final {
    return {dataLabel};
  }

private:
  const NumericType stickingProbability;
  const NumericType exponent;
  const NumericType minAngle;
  const std::string dataLabel = "ionFlux";
};
} // namespace impl

template <class NumericType, int D>
class TEOSPECVD : public ProcessModel<NumericType, D> {
public:
  TEOSPECVD(const NumericType pRadicalSticking, const NumericType pRadicalRate,
            const NumericType pIonRate, const NumericType pIonExponent,
            const NumericType pIonSticking = 1.,
            const NumericType pRadicalOrder = 1.,
            const NumericType pIonOrder = 1.,
            const NumericType pIonMinAngle = 0.) {
    // velocity field
    auto velField = SmartPointer<DefaultVelocityField<NumericType>>::New(2);
    this->setVelocityField(velField);

    // particles
    auto radical = std::make_unique<impl::Radical<NumericType, D>>(
        pRadicalSticking, "radicalFlux");
    auto ion = std::make_unique<impl::Ion<NumericType, D>>(
        pIonSticking, pIonExponent, pIonMinAngle, "ionFlux");

    // surface model
    auto surfModel = SmartPointer<impl::PECVDSurfaceModel<NumericType>>::New(
        pRadicalRate, pRadicalOrder, pIonRate, pIonOrder);

    this->setSurfaceModel(surfModel);
    this->insertNextParticleType(radical);
    this->insertNextParticleType(ion);
    this->setProcessName("TEOSPECVD");
  }
};

} // namespace viennaps