#pragma once

#include "../psMaterials.hpp"
#include "../psProcessModel.hpp"

#include <rayParticle.hpp>
#include <rayReflection.hpp>

#include <functional>
#include <random>

namespace viennaps {

using namespace viennacore;

template <typename NumericType> struct IBEParameters {
  NumericType planeWaferRate = 1.;
  NumericType meanEnergy = 250;     // eV
  NumericType sigmaEnergy = 10;     // eV
  NumericType thresholdEnergy = 20; // eV
  NumericType exponent = 100;
  NumericType n_l = 10;
  NumericType inflectAngle = 89; // degree
  NumericType minAngle = 5;      // degree
  NumericType tiltAngle = 0;     // degree
  std::function<NumericType(NumericType)> yieldFunction =
      [](NumericType theta) { return 1.; };

  // Redeposition
  NumericType redepositionThreshold = 0.1;
  NumericType redepositionRate = 0.0;
};

namespace impl {

template <typename NumericType>
class IBESurfaceModel : public SurfaceModel<NumericType> {
  const IBEParameters<NumericType> params_;
  const std::vector<Material> maskMaterials_;

public:
  IBESurfaceModel(const IBEParameters<NumericType> &params,
                  const std::vector<Material> &mask)
      : maskMaterials_(mask), params_(params) {}

  SmartPointer<std::vector<NumericType>> calculateVelocities(
      SmartPointer<viennals::PointData<NumericType>> rates,
      const std::vector<std::array<NumericType, 3>> &coordinates,
      const std::vector<NumericType> &materialIds) override {

    auto velocity =
        SmartPointer<std::vector<NumericType>>::New(materialIds.size(), 0.);
    auto flux = rates->getScalarData("ionFlux");
    auto redeposition = rates->getScalarData("redepositionFlux");

    const NumericType norm =
        params_.planeWaferRate /
        ((std::sqrt(params_.meanEnergy) - std::sqrt(params_.thresholdEnergy)) *
         params_.yieldFunction(std::cos(params_.tiltAngle * M_PI / 180.)));

    for (std::size_t i = 0; i < velocity->size(); i++) {
      if (!isMaskMaterial(materialIds[i])) {
        velocity->at(i) = -flux->at(i) * norm;
        velocity->at(i) += redeposition->at(i) * params_.redepositionRate;
      }
    }

    return velocity;
  }

private:
  bool isMaskMaterial(const NumericType &material) const {
    for (const auto &mat : maskMaterials_) {
      if (MaterialMap::isMaterial(material, mat))
        return true;
    }
    return false;
  }
};

template <typename NumericType, int D>
class IBEIonWithRedeposition
    : public viennaray::Particle<IBEIonWithRedeposition<NumericType, D>,
                                 NumericType> {
public:
  IBEIonWithRedeposition(const IBEParameters<NumericType> &params)
      : params_(params), normalDist_(params.meanEnergy, params.sigmaEnergy),
        A_(1. / (1. + params.n_l * (M_PI_2 / params.inflectAngle - 1.))),
        inflectAngle_(params.inflectAngle * M_PI / 180.),
        minAngle_(params.minAngle * M_PI / 180.) {}

  void surfaceCollision(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                        const Vec3D<NumericType> &geomNormal,
                        const unsigned int primID, const int,
                        viennaray::TracingData<NumericType> &localData,
                        const viennaray::TracingData<NumericType> *,
                        RNG &) override final {
    auto cosTheta = std::clamp(-DotProduct(rayDir, geomNormal), NumericType(0),
                               NumericType(1));
    NumericType theta = std::acos(cosTheta);

    localData.getVectorData(0)[primID] +=
        std::max(std::sqrt(energy_) - std::sqrt(params_.thresholdEnergy), 0.) *
        params_.yieldFunction(theta);

    if (params_.redepositionRate > 0.)
      localData.getVectorData(1)[primID] += redepositionWeight_;
  }

  std::pair<NumericType, Vec3D<NumericType>>
  surfaceReflection(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                    const Vec3D<NumericType> &geomNormal,
                    const unsigned int primID, const int materialId,
                    const viennaray::TracingData<NumericType> *globalData,
                    RNG &rngState) override final {

    // Small incident angles are reflected with the energy fraction centered at
    // 0
    NumericType incAngle = std::acos(-DotProduct(rayDir, geomNormal));
    NumericType Eref_peak;
    if (incAngle >= inflectAngle_) {
      Eref_peak =
          1. - (1. - A_) * (M_PI_2 - incAngle) / (M_PI_2 - inflectAngle_);
    } else {
      Eref_peak = A_ * std::pow(incAngle / inflectAngle_, params_.n_l);
    }

    if (params_.redepositionRate > 0.) {
      redepositionWeight_ =
          std::max(std::sqrt(energy_) - std::sqrt(params_.thresholdEnergy),
                   0.) *
          params_.yieldFunction(incAngle);
    }

    // Gaussian distribution around the Eref_peak scaled by the particle energy
    NumericType newEnergy;
    std::normal_distribution<NumericType> normalDist(Eref_peak * energy_,
                                                     0.1 * energy_);
    do {
      newEnergy = normalDist(rngState);
    } while (newEnergy > energy_ || newEnergy < 0.);

    if (newEnergy > params_.thresholdEnergy ||
        redepositionWeight_ > params_.redepositionThreshold) {
      energy_ = newEnergy;
      auto direction = viennaray::ReflectionConedCosine<NumericType, D>(
          rayDir, geomNormal, rngState, std::max(incAngle, minAngle_));
      return std::pair<NumericType, Vec3D<NumericType>>{0., direction};
    } else {
      return std::pair<NumericType, Vec3D<NumericType>>{
          1., Vec3D<NumericType>{0., 0., 0.}};
    }
  }

  void initNew(RNG &rngState) override final {
    do {
      energy_ = normalDist_(rngState);
    } while (energy_ < params_.thresholdEnergy);
    redepositionWeight_ = 0.;
  }

  NumericType getSourceDistributionPower() const override final {
    return params_.exponent;
  }

  std::vector<std::string> getLocalDataLabels() const override final {
    return {"ionFlux", "redepositionFlux"};
  }

private:
  NumericType energy_;
  NumericType redepositionWeight_;

  const IBEParameters<NumericType> &params_;
  const NumericType inflectAngle_;
  const NumericType minAngle_;
  const NumericType A_;
  std::normal_distribution<NumericType> normalDist_;
};
} // namespace impl

template <typename NumericType, int D>
class IonBeamEtching : public ProcessModel<NumericType, D> {
public:
  IonBeamEtching() = default;

  IonBeamEtching(const std::vector<Material> &maskMaterial)
      : maskMaterials_(maskMaterial) {}

  IonBeamEtching(const std::vector<Material> &maskMaterial,
                 const IBEParameters<NumericType> &params)
      : maskMaterials_(maskMaterial), params_(params) {}

  IBEParameters<NumericType> &getParameters() { return params_; }

  void setParameters(const IBEParameters<NumericType> &params) {
    params_ = params;
  }

  void initialize(SmartPointer<Domain<NumericType, D>> domain,
                  const NumericType processDuration) override final {
    if (firstInit)
      return;

    // particles
    auto particle =
        std::make_unique<impl::IBEIonWithRedeposition<NumericType, D>>(params_);

    // surface model
    auto surfModel = SmartPointer<impl::IBESurfaceModel<NumericType>>::New(
        params_, maskMaterials_);

    // velocity field
    auto velField = SmartPointer<DefaultVelocityField<NumericType, D>>::New(2);

    this->particles.clear();
    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->particles.clear();
    this->insertNextParticleType(particle);
    this->setProcessName("IonBeamEtching");
    firstInit = true;
  }

  void reset() override final { firstInit = false; }

private:
  bool firstInit = false;
  std::vector<Material> maskMaterials_;
  IBEParameters<NumericType> params_;
};

} // namespace viennaps
