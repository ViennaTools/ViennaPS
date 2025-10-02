#pragma once

#include "../process/psProcessModel.hpp"

#include <rayParticle.hpp>
#include <rayReflection.hpp>
#include <rayUtil.hpp>

namespace viennaps {

namespace impl {
template <class NumericType>
class PECVDSurfaceModel : public SurfaceModel<NumericType> {
  const NumericType radicalRate_;
  const NumericType radicalReactionOrder_;
  const NumericType ionRate_;
  const NumericType ionReactionOrder_;

public:
  PECVDSurfaceModel(NumericType radicalRate, NumericType radicalReactionOrder,
                    NumericType ionRate, NumericType ionReactionOrder)
      : radicalRate_(radicalRate), radicalReactionOrder_(radicalReactionOrder),
        ionRate_(ionRate), ionReactionOrder_(ionReactionOrder) {}

  SmartPointer<std::vector<NumericType>> calculateVelocities(
      SmartPointer<viennals::PointData<NumericType>> rates,
      const std::vector<std::array<NumericType, 3>> &coordinates,
      const std::vector<NumericType> &materialIDs) override {
    // define the surface reaction here
    auto particleFluxRadical = rates->getScalarData("radicalFlux");
    auto particleFluxIon = rates->getScalarData("ionFlux");

    std::vector<NumericType> velocity(particleFluxRadical->size(), 0.);

#pragma omp parallel for
    for (size_t i = 0; i < velocity.size(); i++) {
      // calculate surface velocity based on particle fluxes
      velocity[i] =
          radicalRate_ *
              std::pow(particleFluxRadical->at(i), radicalReactionOrder_) +
          ionRate_ * std::pow(particleFluxIon->at(i), ionReactionOrder_);
    }

    return SmartPointer<std::vector<NumericType>>::New(velocity);
  }
};

template <typename NumericType, int D>
class Ion : public viennaray::Particle<Ion<NumericType, D>, NumericType> {
public:
  Ion(const NumericType stickingProbability, const NumericType exponent,
      const NumericType minAngle)
      : stickingProbability_(stickingProbability), exponent_(exponent),
        minAngle_(minAngle) {}

  std::pair<NumericType, Vec3D<NumericType>>
  surfaceReflection(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                    const Vec3D<NumericType> &geomNormal,
                    const unsigned int primId, const int materialId,
                    const viennaray::TracingData<NumericType> *globalData,
                    RNG &Rng) final {
    auto cosTheta = std::clamp(-DotProduct(rayDir, geomNormal), NumericType(0),
                               NumericType(1));
    NumericType incAngle = std::acos(cosTheta);

    auto direction = viennaray::ReflectionConedCosine<NumericType, D>(
        rayDir, geomNormal, Rng, std::max(incAngle, minAngle_));
    return std::pair<NumericType, Vec3D<NumericType>>{stickingProbability_,
                                                      direction};
  }

  void surfaceCollision(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                        const Vec3D<NumericType> &geomNormal,
                        const unsigned int primID, const int materialId,
                        viennaray::TracingData<NumericType> &localData,
                        const viennaray::TracingData<NumericType> *globalData,
                        RNG &Rng) final {
    localData.getVectorData(0)[primID] += rayWeight;
  }

  NumericType getSourceDistributionPower() const final { return exponent_; }
  [[nodiscard]] std::vector<std::string> getLocalDataLabels() const final {
    return {"ionFlux"};
  }

private:
  const NumericType stickingProbability_;
  const NumericType exponent_;
  const NumericType minAngle_;
};
} // namespace impl

template <class NumericType, int D>
class TEOSPECVD : public ProcessModelCPU<NumericType, D> {
public:
  TEOSPECVD(NumericType radicalSticking, NumericType radicalRate,
            NumericType ionRate, NumericType ionExponent,
            NumericType ionSticking = 1., NumericType radicalOrder = 1.,
            NumericType ionOrder = 1., NumericType ionMinAngle = 0.) {
    // velocity field
    auto velField = SmartPointer<DefaultVelocityField<NumericType, D>>::New();
    this->setVelocityField(velField);

    // particles
    auto radical = std::make_unique<viennaray::DiffuseParticle<NumericType, D>>(
        radicalSticking, "radicalFlux");
    auto ion = std::make_unique<impl::Ion<NumericType, D>>(
        ionSticking, ionExponent, ionMinAngle);

    // surface model
    auto surfModel = SmartPointer<impl::PECVDSurfaceModel<NumericType>>::New(
        radicalRate, radicalOrder, ionRate, ionOrder);

    this->setSurfaceModel(surfModel);
    this->insertNextParticleType(radical);
    this->insertNextParticleType(ion);
    this->setProcessName("TEOSPECVD");

    this->processMetaData["RadicalSticking"] = {radicalSticking};
    this->processMetaData["RadicalRate"] = {radicalRate};
    this->processMetaData["RadicalOrder"] = {radicalOrder};
    this->processMetaData["IonRate"] = {ionRate};
    this->processMetaData["IonSticking"] = {ionSticking};
    this->processMetaData["IonExponent"] = {ionExponent};
    this->processMetaData["IonOrder"] = {ionOrder};
    this->processMetaData["IonMinAngle"] = {ionMinAngle};
  }
};

PS_PRECOMPILE_PRECISION_DIMENSION(TEOSPECVD)

} // namespace viennaps