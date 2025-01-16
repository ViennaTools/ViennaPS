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
  NumericType n = 10;
  NumericType inflectAngle = 89; // degree
  NumericType minAngle = 5;      // degree
  NumericType tiltAngle = 0;     // degree
  std::function<NumericType(NumericType)> yieldFunction =
      [](NumericType theta) { return 1.; };
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
      if (MaterialMap::isMaterial(material, mat))
        return true;
    }
    return false;
  }
};

template <typename NumericType, int D>
class IBEIon : public viennaray::Particle<IBEIon<NumericType, D>, NumericType> {
public:
  IBEIon(const IBEParameters<NumericType> &params)
      : params_(params), normalDist_(params.meanEnergy, params.sigmaEnergy),
        A_(1. / (1. + params.n * (M_PI_2 / params.inflectAngle - 1.))),
        inflectAngle_(params.inflectAngle * M_PI / 180.),
        minAngle_(params.minAngle * M_PI / 180.) {}

  void surfaceCollision(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                        const Vec3D<NumericType> &geomNormal,
                        const unsigned int primID, const int,
                        viennaray::TracingData<NumericType> &localData,
                        const viennaray::TracingData<NumericType> *,
                        RNG &) override final {
    auto cosTheta = -DotProduct(rayDir, geomNormal);
    NumericType theta =
        std::acos(std::max(std::min(cosTheta, static_cast<NumericType>(1.)),
                           static_cast<NumericType>(0.)));

    localData.getVectorData(0)[primID] +=
        std::max(std::sqrt(energy_) - std::sqrt(params_.thresholdEnergy), 0.) *
        params_.yieldFunction(theta);
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
      Eref_peak = A_ * std::pow(incAngle / inflectAngle_, params_.n);
    }
    // Gaussian distribution around the Eref_peak scaled by the particle energy
    NumericType newEnergy;
    std::normal_distribution<NumericType> normalDist(Eref_peak * energy_,
                                                     0.1 * energy_);
    do {
      newEnergy = normalDist(rngState);
    } while (newEnergy > energy_ || newEnergy < 0.);

    if (newEnergy > params_.thresholdEnergy) {
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
  }

  NumericType getSourceDistributionPower() const override final {
    return params_.exponent;
  }

  std::vector<std::string> getLocalDataLabels() const override final {
    return {"ionFlux"};
  }

private:
  NumericType energy_;

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
  IonBeamEtching() { initialize(maskMaterials_); }

  IonBeamEtching(std::vector<Material> maskMaterial) {
    maskMaterials_ = std::move(maskMaterial);
    initialize(maskMaterials_);
  }

  IBEParameters<NumericType> &getParameters() { return params_; }

  void setParameters(const IBEParameters<NumericType> &params) {
    params_ = params;
    initialize(maskMaterials_);
  }

private:
  void initialize(const std::vector<Material> &maskMaterial) {
    // particles
    auto particle = std::make_unique<impl::IBEIon<NumericType, D>>(params_);

    // surface model
    auto surfModel = SmartPointer<impl::IBESurfaceModel<NumericType>>::New(
        params_, maskMaterial);

    // velocity field
    auto velField = SmartPointer<DefaultVelocityField<NumericType, D>>::New(2);

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->particles.clear();
    this->insertNextParticleType(particle);
    this->setProcessName("IonBeamEtching");
  }

private:
  IBEParameters<NumericType> params_;
  std::vector<Material> maskMaterials_;
};

} // namespace viennaps
