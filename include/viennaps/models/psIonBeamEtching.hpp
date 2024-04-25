#pragma once

#include "../psMaterials.hpp"
#include "../psProcessModel.hpp"

#include <rayParticle.hpp>
#include <rayReflection.hpp>

#include <random>

namespace IBEImplementation {
template <typename NumericType>
class SurfaceModel : public psSurfaceModel<NumericType> {
  const std::vector<psMaterial> maskMaterials_;
  const NumericType planeWaferRate_;
  const NumericType meanEnergy_;
  const NumericType thresholdEnergy_;
  const NumericType tiltAngle_;
  const std::function<NumericType(NumericType)> &yieldFunction_;

public:
  SurfaceModel(NumericType rate, const std::vector<psMaterial> &mask,
               NumericType meanEnergy, NumericType thresholdEnergy,
               NumericType tiltAngle,
               const std::function<NumericType(NumericType)> &yieldFunction)
      : maskMaterials_(mask), planeWaferRate_(rate), meanEnergy_(meanEnergy),
        thresholdEnergy_(thresholdEnergy), tiltAngle_(tiltAngle),
        yieldFunction_(yieldFunction) {}

  psSmartPointer<std::vector<NumericType>> calculateVelocities(
      psSmartPointer<psPointData<NumericType>> rates,
      const std::vector<std::array<NumericType, 3>> &coordinates,
      const std::vector<NumericType> &materialIds) override {

    auto velocity =
        psSmartPointer<std::vector<NumericType>>::New(materialIds.size(), 0.);
    auto flux = rates->getScalarData("particleFlux");

    const NumericType norm =
        planeWaferRate_ /
        ((std::sqrt(meanEnergy_) - std::sqrt(thresholdEnergy_)) *
         yieldFunction_(std::cos(tiltAngle_)));

    for (std::size_t i = 0; i < velocity->size(); i++) {
      if (!isMaskMaterial(materialIds[i])) {
        velocity->at(i) = flux->at(i) * norm;
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
class Ion : public rayParticle<Ion<NumericType, D>, NumericType> {
public:
  Ion(NumericType meanEnergy, NumericType sigmaEnergy,
      NumericType thresholdEnergy, NumericType sourcePower, NumericType n,
      NumericType inflectAngle, NumericType minAngle,
      const std::function<NumericType(NumericType)> &yieldFunction)
      : meanEnergy_(meanEnergy), sigmaEnergy_(sigmaEnergy),
        thresholdEnergy_(thresholdEnergy), sourcePower_(sourcePower),
        normalDist_(meanEnergy, sigmaEnergy), n_(n),
        inflectAngle_(inflectAngle), minAngle_(minAngle),
        yieldFunction_(yieldFunction),
        A_(1. / (1. + n * (M_PI_2 / inflectAngle - 1.))) {}

  void surfaceCollision(NumericType rayWeight,
                        const rayTriple<NumericType> &rayDir,
                        const rayTriple<NumericType> &geomNormal,
                        const unsigned int primID, const int materialId,
                        rayTracingData<NumericType> &localData,
                        const rayTracingData<NumericType> *globalData,
                        rayRNG &Rng) override final {
    NumericType cosTheta = -rayInternal::DotProduct(rayDir, geomNormal);

    localData.getVectorData(0)[primID] +=
        std::max(std::sqrt(energy_) - std::sqrt(thresholdEnergy_), 0.) *
        yieldFunction_(cosTheta);
  }

  std::pair<NumericType, rayTriple<NumericType>>
  surfaceReflection(NumericType rayWeight, const rayTriple<NumericType> &rayDir,
                    const rayTriple<NumericType> &geomNormal,
                    const unsigned int primID, const int materialId,
                    const rayTracingData<NumericType> *globalData,
                    rayRNG &Rng) override final {

    // Small incident angles are reflected with the energy fraction centered at
    // 0
    NumericType incAngle =
        std::acos(-rayInternal::DotProduct(rayDir, geomNormal));
    NumericType Eref_peak;
    if (incAngle >= inflectAngle_) {
      Eref_peak =
          1. - (1. - A_) * (M_PI_2 - incAngle) / (M_PI_2 - inflectAngle_);
    } else {
      Eref_peak = A_ * std::pow(incAngle / inflectAngle_, n_);
    }
    // Gaussian distribution around the Eref_peak scaled by the particle energy
    NumericType newEnergy;
    std::normal_distribution<NumericType> normalDist(Eref_peak * energy_,
                                                     0.1 * energy_);
    do {
      newEnergy = normalDist(Rng);
    } while (newEnergy > energy_ || newEnergy < 0.);

    if (newEnergy > thresholdEnergy_) {
      energy_ = newEnergy;
      auto direction = rayReflectionConedCosine<NumericType, D>(
          rayDir, geomNormal, Rng, std::max(incAngle, minAngle_));
      return std::pair<NumericType, rayTriple<NumericType>>{0., direction};
    } else {
      return std::pair<NumericType, rayTriple<NumericType>>{
          1., rayTriple<NumericType>{0., 0., 0.}};
    }
  }

  void initNew(rayRNG &RNG) override final {
    do {
      energy_ = normalDist_(RNG);
    } while (energy_ < thresholdEnergy_);
  }

  NumericType getSourceDistributionPower() const override final {
    return sourcePower_;
  }

  std::vector<std::string> getLocalDataLabels() const override final {
    return {"ionFlux"};
  }

private:
  NumericType energy_;

  const NumericType thresholdEnergy_;

  const NumericType meanEnergy_;
  const NumericType sigmaEnergy_;

  const NumericType sourcePower_;
  const std::normal_distribution<NumericType> normalDist_;

  const NumericType n_;
  const NumericType inflectAngle_;
  const NumericType minAngle_;
  const NumericType A_;

  const std::function<NumericType(NumericType)> &yieldFunction_;
};
} // namespace IBEImplementation

template <typename NumericType, int D>
class psIonBeamEtching : public psProcessModel<NumericType, D> {
public:
  psIonBeamEtching() {}

private:
  void initialize(std::vector<psMaterial> &&maskMaterial) {
    // particles
    auto particle = std::make_unique<IBEImplementation::Ion<NumericType, D>>();

    // surface model
    auto surfModel =
        psSmartPointer<IBEImplementation::SurfaceModel<NumericType>>::New();

    // velocity field
    auto velField = psSmartPointer<psDefaultVelocityField<NumericType>>::New(2);

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->insertNextParticleType(particle);
    this->setProcessName("IonBeamEtching");
  }
};
