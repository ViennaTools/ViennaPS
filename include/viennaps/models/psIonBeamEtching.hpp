#pragma once

#include "../psMaterials.hpp"
#include "../psProcessModel.hpp"
#include "psIonBeamParameters.hpp"

#include <rayParticle.hpp>
#include <rayReflection.hpp>

#include <functional>
#include <random>

#ifdef VIENNATOOLS_PYTHON_BUILD
#include <Python.h>
#endif

namespace viennaps {

using namespace viennacore;

namespace impl {
template <typename NumericType>
class IBESurfaceModel : public SurfaceModel<NumericType> {
  const IBEParameters<NumericType> params_;
  const std::vector<Material> maskMaterials_;

public:
  IBESurfaceModel(const IBEParameters<NumericType> &params,
                  const std::vector<Material> &mask)
      : params_(params), maskMaterials_(mask) {}

  SmartPointer<std::vector<NumericType>>
  calculateVelocities(SmartPointer<viennals::PointData<NumericType>> rates,
                      const std::vector<Vec3D<NumericType>> &coordinates,
                      const std::vector<NumericType> &materialIds) override {

    auto velocity =
        SmartPointer<std::vector<NumericType>>::New(materialIds.size(), 0.);
    auto flux = rates->getScalarData("ionFlux");
    auto redeposition = rates->getScalarData("redepositionFlux");

    const NumericType norm =
        1. /
        ((std::sqrt(params_.meanEnergy) - std::sqrt(params_.thresholdEnergy)) *
         params_.yieldFunction(std::cos(params_.tiltAngle * M_PI / 180.)));

#pragma omp parallel for
    for (size_t i = 0; i < velocity->size(); i++) {
      if (!isMaskMaterial(materialIds[i])) {
        NumericType rate = params_.planeWaferRate;
        if (auto material = MaterialMap::mapToMaterial(materialIds[i]);
            params_.materialPlaneWaferRate.find(material) !=
            params_.materialPlaneWaferRate.end()) {
          rate = params_.materialPlaneWaferRate.at(material);
        }

        velocity->at(i) = -flux->at(i) * norm * rate;
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
class IBEIonWithRedeposition final
    : public viennaray::Particle<IBEIonWithRedeposition<NumericType, D>,
                                 NumericType> {
public:
  explicit IBEIonWithRedeposition(const IBEParameters<NumericType> &params)
      : params_(params), inflectAngle_(params.inflectAngle * M_PI / 180.),
        minAngle_(params.minAngle * M_PI / 180.),
        A_(1. / (1. + params.n_l * (M_PI_2 / params.inflectAngle - 1.))),
        normalDist_(params.meanEnergy, params.sigmaEnergy) {}

  void surfaceCollision(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                        const Vec3D<NumericType> &geomNormal,
                        const unsigned int primID, const int,
                        viennaray::TracingData<NumericType> &localData,
                        const viennaray::TracingData<NumericType> *,
                        RNG &) override {
    auto cosTheta = std::clamp(-DotProduct(rayDir, geomNormal), NumericType(0),
                               NumericType(1));
    NumericType theta = std::acos(cosTheta);

    NumericType yield = params_.yieldFunction(theta);

    localData.getVectorData(0)[primID] +=
        std::max(std::sqrt(energy_) - std::sqrt(params_.thresholdEnergy),
                 NumericType(0.)) *
        yield;

    if (params_.redepositionRate > 0.)
      localData.getVectorData(1)[primID] += redepositionWeight_;
  }

  std::pair<NumericType, Vec3D<NumericType>>
  surfaceReflection(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                    const Vec3D<NumericType> &geomNormal,
                    const unsigned int primID, const int materialId,
                    const viennaray::TracingData<NumericType> *globalData,
                    RNG &rngState) override {

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
                   NumericType(0.)) *
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
          rayDir, geomNormal, rngState, M_PI_2 - std::min(incAngle, minAngle_));
      return std::pair<NumericType, Vec3D<NumericType>>{0., direction};
    } else {
      return std::pair<NumericType, Vec3D<NumericType>>{
          1., Vec3D<NumericType>{0., 0., 0.}};
    }
  }

  void initNew(RNG &rngState) override {
    do {
      energy_ = normalDist_(rngState);
    } while (energy_ < params_.thresholdEnergy);
    redepositionWeight_ = 0.;
  }

  NumericType getSourceDistributionPower() const override {
    return params_.exponent;
  }

  std::vector<std::string> getLocalDataLabels() const override {
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

  explicit IonBeamEtching(const std::vector<Material> &maskMaterial)
      : maskMaterials_(maskMaterial) {}

  IonBeamEtching(const std::vector<Material> &maskMaterial,
                 const IBEParameters<NumericType> &params)
      : maskMaterials_(maskMaterial), params_(params) {}

  IBEParameters<NumericType> &getParameters() { return params_; }

  void setParameters(const IBEParameters<NumericType> &params) {
    params_ = params;
  }

  void initialize(SmartPointer<Domain<NumericType, D>> domain,
                  const NumericType processDuration) final {
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
    this->processMetaData = params_.toProcessMetaData();
    firstInit = true;

    if (Logger::getLogLevel() >= static_cast<unsigned>(LogLevel::DEBUG)) {
      std::stringstream ss;
      ss << "Initialized IonBeamEtching with parameters:\n"
         << "\tPlane wafer rate: " << params_.planeWaferRate << "\n"
         << "\tMaterial plane wafer rate:\n";
      for (const auto &[mat, rate] : params_.materialPlaneWaferRate) {
        ss << "\t    " << MaterialMap::getMaterialName(mat) << ": " << rate
           << "\n";
      }
      ss << "\tMean energy: " << params_.meanEnergy << " eV\n"
         << "\tSigma energy: " << params_.sigmaEnergy << " eV\n"
         << "\tThreshold energy: " << params_.thresholdEnergy << " eV\n"
         << "\tExponent: " << params_.exponent << "\n"
         << "\tn_l: " << params_.n_l << "\n"
         << "\tInflection angle: " << params_.inflectAngle << " degree\n"
         << "\tMinimum angle: " << params_.minAngle << " degree\n"
         << "\tTilt angle: " << params_.tiltAngle << " degree\n"
         << "\tRedeposition threshold: " << params_.redepositionThreshold
         << "\n"
         << "\tRedeposition rate: " << params_.redepositionRate << "\n"
         << "\tMask materials:\n";
      for (const auto &mat : maskMaterials_) {
        ss << "\t    " << MaterialMap::getMaterialName(mat) << "\n";
      }
      Logger::getInstance().addDebug(ss.str()).print();
    }
  }

  void finalize(SmartPointer<Domain<NumericType, D>> domain,
                const NumericType processedDuration) final {
    firstInit = false;
  }

private:
  bool firstInit = false;
  std::vector<Material> maskMaterials_;
  IBEParameters<NumericType> params_;
};

PS_PRECOMPILE_PRECISION_DIMENSION(IonBeamEtching)

} // namespace viennaps
