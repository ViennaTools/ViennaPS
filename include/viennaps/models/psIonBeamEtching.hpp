#pragma once

#include "../process/psProcessModel.hpp"
#include "../psConstants.hpp"
#include "../psMaterials.hpp"
#include "psIonBeamParameters.hpp"

#include <rayParticle.hpp>
#include <rayReflection.hpp>

#include <functional>
#include <random>

#ifdef VIENNACORE_COMPILE_GPU
#include <models/psgPipelineParameters.hpp>
#include <raygCallableConfig.hpp>
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
    std::vector<NumericType> redeposition(materialIds.size(), 0.);
    if (params_.redepositionRate > 0.) {
      redeposition = *rates->getScalarData("redepositionFlux");
    }

    NumericType yield;
    NumericType theta = constants::degToRad(params_.tiltAngle);
    NumericType cosTheta = std::cos(theta);
    if (params_.cos4Yield.isDefined) {
      NumericType cosTheta2 = cosTheta * cosTheta;
      yield =
          (params_.cos4Yield.a1 * cosTheta + params_.cos4Yield.a2 * cosTheta2 +
           params_.cos4Yield.a3 * cosTheta2 * cosTheta +
           params_.cos4Yield.a4 * cosTheta2 * cosTheta2) /
          params_.cos4Yield.aSum();
    } else {
      yield = params_.yieldFunction(theta);
    }

    const NumericType norm =
        1. /
        ((std::sqrt(params_.meanEnergy) - std::sqrt(params_.thresholdEnergy)) *
         yield);

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
        velocity->at(i) += redeposition.at(i) * params_.redepositionRate;
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
        sqrtThresholdEnergy_(std::sqrt(params.thresholdEnergy)),
        thetaRMin_(params.thetaRMin * M_PI / 180.),
        thetaRMax_(params.thetaRMax * M_PI / 180.),
        aSum_(1. / params.cos4Yield.aSum()),
        normalDist_(params.meanEnergy, params.sigmaEnergy) {}

  void surfaceCollision(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                        const Vec3D<NumericType> &geomNormal,
                        const unsigned int primID, const int,
                        viennaray::TracingData<NumericType> &localData,
                        const viennaray::TracingData<NumericType> *,
                        RNG &) override {
    auto cosTheta = std::clamp(-DotProduct(rayDir, geomNormal), NumericType(0),
                               NumericType(1));
    NumericType yield;
    if (params_.cos4Yield.isDefined) {
      NumericType cosTheta2 = cosTheta * cosTheta;
      yield =
          (params_.cos4Yield.a1 * cosTheta + params_.cos4Yield.a2 * cosTheta2 +
           params_.cos4Yield.a3 * cosTheta2 * cosTheta +
           params_.cos4Yield.a4 * cosTheta2 * cosTheta2) *
          aSum_;
    } else {
      yield = params_.yieldFunction(std::acos(cosTheta));
    }

    localData.getVectorData(0)[primID] +=
        std::max(std::sqrt(energy_) - sqrtThresholdEnergy_, NumericType(0)) *
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

    const NumericType cosTheta = std::clamp(-DotProduct(rayDir, geomNormal),
                                            NumericType(0), NumericType(1));
    const NumericType theta = std::acos(cosTheta);

    // Update redeposition weight
    if (params_.redepositionRate > 0.) {
      NumericType yield;
      if (params_.cos4Yield.isDefined) {
        NumericType cosTheta2 = cosTheta * cosTheta;
        yield = (params_.cos4Yield.a1 * cosTheta +
                 params_.cos4Yield.a2 * cosTheta2 +
                 params_.cos4Yield.a3 * cosTheta2 * cosTheta +
                 params_.cos4Yield.a4 * cosTheta2 * cosTheta2) *
                aSum_;
      } else {
        yield = params_.yieldFunction(std::acos(cosTheta));
      }
      redepositionWeight_ =
          std::max(std::sqrt(energy_) - sqrtThresholdEnergy_, NumericType(0)) *
          yield;
    }

    NumericType sticking = 1.;
    if (theta > params_.thetaRMin) {
      sticking = 1. - std::clamp((theta - params_.thetaRMin) /
                                     (params_.thetaRMax - params_.thetaRMin),
                                 NumericType(0), NumericType(1));
    }

    // Early exit: particle sticks and no redeposition
    if (sticking >= 1. && redepositionWeight_ < params_.redepositionThreshold) {
      return VIENNARAY_PARTICLE_STOP;
    }

    // Calculate new energy after reflection
    NumericType Eref_peak;
    if (theta >= inflectAngle_) {
      Eref_peak = 1. - (1. - A_) * (M_PI_2 - theta) / (M_PI_2 - inflectAngle_);
    } else {
      Eref_peak = A_ * std::pow(theta / inflectAngle_, params_.n_l);
    }

    // Gaussian distribution around the Eref_peak scaled by the particle
    // energy
    NumericType newEnergy = Eref_peak * energy_;
    std::normal_distribution<NumericType> normalDist(Eref_peak * energy_,
                                                     0.1 * energy_);
    do {
      newEnergy = normalDist(rngState);
    } while (newEnergy > energy_ || newEnergy < 0.);

    if (newEnergy > params_.thresholdEnergy ||
        redepositionWeight_ > params_.redepositionThreshold) {
      energy_ = newEnergy;
      auto direction = viennaray::ReflectionConedCosine<NumericType, D>(
          rayDir, geomNormal, rngState, M_PI_2 - std::min(theta, minAngle_));
      return std::pair<NumericType, Vec3D<NumericType>>{sticking, direction};
    } else {
      return VIENNARAY_PARTICLE_STOP;
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
  const NumericType inflectAngle_; // in rad
  const NumericType minAngle_;     // in rad
  const NumericType A_;
  const NumericType sqrtThresholdEnergy_;
  const NumericType thetaRMin_; // in rad
  const NumericType thetaRMax_; // in  rad
  const NumericType aSum_;

  std::normal_distribution<NumericType> normalDist_;
};
} // namespace impl

#ifdef VIENNACORE_COMPILE_GPU
namespace gpu {

template <typename NumericType, int D>
class IonBeamEtching : public ProcessModelGPU<NumericType, D> {
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

    if (params_.redepositionRate > 0.) {
      Logger::getInstance()
          .addWarning(
              "IonBeamEtching GPU process does not support redeposition. "
              "Redeosition parameters will be ignored.")
          .print();
      params_.redepositionRate = 0.;
    }

    // particles
    viennaray::gpu::Particle<NumericType> particle{
        .name = "IBEIon", .cosineExponent = params_.exponent};
    particle.dataLabels.push_back("ionFlux");

    if (params_.tiltAngle != 0.) {
      Vec3D<NumericType> direction{0., 0., 0.};
      direction[D - 2] = std::sin(constants::degToRad(params_.tiltAngle));
      direction[D - 1] = -std::cos(constants::degToRad(params_.tiltAngle));
      particle.direction = direction;
      particle.useCustomDirection = true;
    }

    std::unordered_map<std::string, unsigned> pMap = {{"IBEIon", 0}};
    std::vector<viennaray::gpu::CallableConfig> cMap = {
        {0, viennaray::gpu::CallableSlot::COLLISION,
         "__direct_callable__IBECollision"},
        {0, viennaray::gpu::CallableSlot::REFLECTION,
         "__direct_callable__IBEReflection"},
        {0, viennaray::gpu::CallableSlot::INIT, "__direct_callable__IBEInit"}};
    this->setParticleCallableMap(pMap, cMap);
    this->setCallableFileName("CallableWrapper");

    impl::IonParams deviceParams;
    deviceParams.thetaRMin =
        static_cast<float>(constants::degToRad(params_.thetaRMin));
    deviceParams.thetaRMax =
        static_cast<float>(constants::degToRad(params_.thetaRMax));
    deviceParams.meanEnergy = static_cast<float>(params_.meanEnergy);
    deviceParams.sigmaEnergy = static_cast<float>(params_.sigmaEnergy);
    deviceParams.thresholdEnergy = static_cast<float>(
        std::sqrt(params_.thresholdEnergy)); // precompute sqrt
    deviceParams.minAngle =
        static_cast<float>(constants::degToRad(params_.minAngle));
    deviceParams.inflectAngle =
        static_cast<float>(constants::degToRad(params_.inflectAngle));
    deviceParams.n_l = static_cast<float>(params_.n_l);
    deviceParams.B_sp = 0.f; // not used in IBE
    if (params_.cos4Yield.isDefined) {
      deviceParams.a1 = static_cast<float>(params_.cos4Yield.a1);
      deviceParams.a2 = static_cast<float>(params_.cos4Yield.a2);
      deviceParams.a3 = static_cast<float>(params_.cos4Yield.a3);
      deviceParams.a4 = static_cast<float>(params_.cos4Yield.a4);
      deviceParams.aSum = static_cast<float>(params_.cos4Yield.aSum());
    }

    // upload process params
    this->processData.alloc(sizeof(impl::IonParams));
    this->processData.upload(&deviceParams, 1);

    // surface model
    auto surfModel =
        SmartPointer<::viennaps::impl::IBESurfaceModel<NumericType>>::New(
            params_, maskMaterials_);

    // velocity field
    auto velField = SmartPointer<DefaultVelocityField<NumericType, D>>::New();

    this->getParticleTypes().clear();
    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->insertNextParticleType(particle);
    this->setProcessName("IonBeamEtching");
    this->processMetaData = params_.toProcessMetaData();
    this->hasGPU = true;

    if (Logger::getLogLevel() >= static_cast<unsigned>(LogLevel::DEBUG)) {
      Logger::getInstance()
          .addDebug("Process parameters:" +
                    util::metaDataToString(this->processMetaData))
          .print();
    }

    firstInit = true;
  }

  void finalize(SmartPointer<Domain<NumericType, D>> domain,
                const NumericType processedDuration) final {
    firstInit = false;
  }

  bool useFluxEngine() final { return true; }

private:
  bool firstInit = false;
  std::vector<Material> maskMaterials_;
  IBEParameters<NumericType> params_;
};

} // namespace gpu
#endif

template <typename NumericType, int D>
class IonBeamEtching : public ProcessModelCPU<NumericType, D> {
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
    auto velField = SmartPointer<DefaultVelocityField<NumericType, D>>::New();

    this->particles.clear();
    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->insertNextParticleType(particle);
    this->setProcessName("IonBeamEtching");
    this->processMetaData = params_.toProcessMetaData();
    this->hasGPU = true;

    if (Logger::getLogLevel() >= static_cast<unsigned>(LogLevel::DEBUG)) {
      Logger::getInstance()
          .addDebug("Process parameters:" +
                    util::metaDataToString(this->processMetaData))
          .print();
    }

    firstInit = true;
  }

  void finalize(SmartPointer<Domain<NumericType, D>> domain,
                const NumericType processedDuration) final {
    firstInit = false;
  }

  bool useFluxEngine() final { return true; }

#ifdef VIENNACORE_COMPILE_GPU
  SmartPointer<ProcessModelBase<NumericType, D>> getGPUModel() final {
    return SmartPointer<gpu::IonBeamEtching<NumericType, D>>::New(
        maskMaterials_, params_);
  }
#endif

private:
  bool firstInit = false;
  std::vector<Material> maskMaterials_;
  IBEParameters<NumericType> params_;
};

PS_PRECOMPILE_PRECISION_DIMENSION(IonBeamEtching)

} // namespace viennaps
