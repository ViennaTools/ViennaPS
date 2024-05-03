#pragma once

#include "../psMaterials.hpp"
#include "../psProcessModel.hpp"

#include <rayParticle.hpp>
#include <rayReflection.hpp>

#include <random>

namespace IBEImplementation {
template <typename NumericType> struct Parameters {
  NumericType planeWaferRate = 1.;
  NumericType meanEnergy = 250;     // eV
  NumericType sigmaEnergy = 10;     // eV
  NumericType thresholdEnergy = 20; // eV
  NumericType sourcePower = 100;
  NumericType n = 10;
  NumericType inflectAngle = 89; // degree
  NumericType minAngle = 5;      // degree
  NumericType tiltAngle = 0;     // degree
  std::function<NumericType(NumericType)> yieldFunction =
      [](NumericType cosTheta) { return 1.; };
};

template <typename NumericType>
class SurfaceModel : public psSurfaceModel<NumericType> {
  const Parameters<NumericType> params_;
  const std::vector<psMaterial> maskMaterials_;

public:
  SurfaceModel(const Parameters<NumericType> &params,
               const std::vector<psMaterial> &mask)
      : maskMaterials_(mask), params_(params) {}

  psSmartPointer<std::vector<NumericType>> calculateVelocities(
      psSmartPointer<psPointData<NumericType>> rates,
      const std::vector<std::array<NumericType, 3>> &coordinates,
      const std::vector<NumericType> &materialIds) override {

    auto velocity =
        psSmartPointer<std::vector<NumericType>>::New(materialIds.size(), 0.);
    auto flux = rates->getScalarData("ionFlux");

    const NumericType norm =
        params_.planeWaferRate /
        ((std::sqrt(params_.meanEnergy) - std::sqrt(params_.thresholdEnergy)) *
         params_.yieldFunction(std::cos(params_.tiltAngle * M_PI / 180.)));

    for (std::size_t i = 0; i < velocity->size(); i++) {
      if (!isMaskMaterial(materialIds[i])) {
        velocity->at(i) = -flux->at(i) * norm;
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
  Ion(const Parameters<NumericType> &params)
      : params_(params), normalDist_(params.meanEnergy, params.sigmaEnergy),
        A_(1. / (1. + params.n * (M_PI_2 / params.inflectAngle - 1.))),
        inflectAngle_(params.inflectAngle * M_PI / 180.),
        minAngle_(params.minAngle * M_PI / 180.) {}

  void surfaceCollision(NumericType rayWeight,
                        const rayTriple<NumericType> &rayDir,
                        const rayTriple<NumericType> &geomNormal,
                        const unsigned int primID, const int materialId,
                        rayTracingData<NumericType> &localData,
                        const rayTracingData<NumericType> *globalData,
                        rayRNG &Rng) override final {
    NumericType cosTheta = -rayInternal::DotProduct(rayDir, geomNormal);

    localData.getVectorData(0)[primID] +=
        std::max(std::sqrt(energy_) - std::sqrt(params_.thresholdEnergy), 0.) *
        params_.yieldFunction(cosTheta);
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
      Eref_peak = A_ * std::pow(incAngle / inflectAngle_, params_.n);
    }
    // Gaussian distribution around the Eref_peak scaled by the particle energy
    NumericType newEnergy;
    std::normal_distribution<NumericType> normalDist(Eref_peak * energy_,
                                                     0.1 * energy_);
    do {
      newEnergy = normalDist(Rng);
    } while (newEnergy > energy_ || newEnergy < 0.);

    if (newEnergy > params_.thresholdEnergy) {
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
    } while (energy_ < params_.thresholdEnergy);
  }

  NumericType getSourceDistributionPower() const override final {
    return params_.sourcePower;
  }

  std::vector<std::string> getLocalDataLabels() const override final {
    return {"ionFlux"};
  }

private:
  NumericType energy_;

  const Parameters<NumericType> &params_;
  const NumericType inflectAngle_;
  const NumericType minAngle_;
  const NumericType A_;
  std::normal_distribution<NumericType> normalDist_;
};
} // namespace IBEImplementation

template <typename NumericType, int D>
class psIonBeamEtching : public psProcessModel<NumericType, D> {
public:
  psIonBeamEtching() {
    std::vector<psMaterial> maskMaterial;
    initialize(std::move(maskMaterial));
  }

  psIonBeamEtching(std::vector<psMaterial> maskMaterial) {
    initialize(std::move(maskMaterial));
  }

  IBEImplementation::Parameters<NumericType> &getParameters() {
    return params_;
  }

  void setParameters(const IBEImplementation::Parameters<NumericType> &params) {
    params_ = params;
  }

private:
  void initialize(std::vector<psMaterial> &&maskMaterial) {
    // particles
    auto particle =
        std::make_unique<IBEImplementation::Ion<NumericType, D>>(params_);

    // surface model
    auto surfModel =
        psSmartPointer<IBEImplementation::SurfaceModel<NumericType>>::New(
            params_, maskMaterial);

    // velocity field
    auto velField = psSmartPointer<psDefaultVelocityField<NumericType>>::New(2);

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->insertNextParticleType(particle);
    this->setProcessName("IonBeamEtching");
  }

private:
  IBEImplementation::Parameters<NumericType> params_;
};
