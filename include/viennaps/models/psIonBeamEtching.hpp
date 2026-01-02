#pragma once

#include "../process/psProcessModel.hpp"
#include "../psConstants.hpp"
#include "../psMaterials.hpp"
#include "psIonBeamParameters.hpp"
#include "psIonModelUtil.hpp"
#include "psPipelineParameters.hpp"

#include <rayParticle.hpp>
#include <rayReflection.hpp>

#ifdef VIENNACORE_COMPILE_GPU
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
  constexpr static const char *fluxLabel = "ionFlux";
  constexpr static const char *redepositionLabel = "redepositionFlux";

  IBESurfaceModel(const IBEParameters<NumericType> &params,
                  const std::vector<Material> &mask)
      : params_(params), maskMaterials_(mask) {}

  SmartPointer<std::vector<NumericType>>
  calculateVelocities(SmartPointer<viennals::PointData<NumericType>> rates,
                      const std::vector<Vec3D<NumericType>> &coordinates,
                      const std::vector<NumericType> &materialIds) override {

    auto velocity =
        SmartPointer<std::vector<NumericType>>::New(materialIds.size(), 0.);
    auto flux = rates->getScalarData(fluxLabel);
    std::vector<NumericType> redeposition(materialIds.size(), 0.);
    if (params_.redepositionRate > 0.) {
      redeposition = *rates->getScalarData(redepositionLabel);
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
      : params_(params),
        inflectAngle_(constants::degToRad(params.inflectAngle)),
        minAngle_(constants::degToRad(params.minAngle)),
        A_(1. / (1. + params.n_l * (M_PI_2 / inflectAngle_ - 1.))),
        sqrtThresholdEnergy_(std::sqrt(params.thresholdEnergy)),
        thetaRMin_(constants::degToRad(params.thetaRMin)),
        thetaRMax_(constants::degToRad(params.thetaRMax)),
        aSum_(1. / params.cos4Yield.aSum()),
        tiltAngle_(constants::degToRad(params.tiltAngle)) {}

  void surfaceCollision(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                        const Vec3D<NumericType> &geomNormal,
                        const unsigned int primID, const int,
                        viennaray::TracingData<NumericType> &localData,
                        const viennaray::TracingData<NumericType> *,
                        RNG &) override {
    auto cosTheta = util::saturate(-DotProduct(rayDir, geomNormal));
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
        rayWeight *
        std::max(std::sqrt(energy_) - sqrtThresholdEnergy_, NumericType(0)) *
        yield;

    if (params_.redepositionRate > 0. && redepositionWeight_ > 0.)
      localData.getVectorData(1)[primID] += redepositionWeight_;
  }

  std::pair<NumericType, Vec3D<NumericType>>
  surfaceReflection(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                    const Vec3D<NumericType> &geomNormal,
                    const unsigned int primID, const int materialId,
                    const viennaray::TracingData<NumericType> *globalData,
                    RNG &rngState) override {

    const NumericType cosTheta = getCosTheta(rayDir, geomNormal);
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
        yield = params_.yieldFunction(theta);
      }
      redepositionWeight_ =
          std::max(std::sqrt(energy_) - sqrtThresholdEnergy_, NumericType(0)) *
          yield;
    }

    NumericType sticking = 1.;
    if (theta > thetaRMin_) {
      sticking =
          1. - util::saturate((theta - thetaRMin_) / (thetaRMax_ - thetaRMin_));
    }

    // Early exit: particle sticks and no redeposition
    if (sticking >= 1. && redepositionWeight_ <= 0.) {
      return VIENNARAY_PARTICLE_STOP;
    }

    // Calculate new energy after reflection
    NumericType newEnergy =
        updateEnergy(rngState, energy_, theta, A_, inflectAngle_, params_.n_l);

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
    energy_ =
        initNormalDistEnergy(rngState, params_.meanEnergy, params_.sigmaEnergy);
    redepositionWeight_ = 0.;
  }

  Vec3D<NumericType> initNewWithDirection(RNG &rngState) override {
    if (params_.rotatingWafer) {
      // 1) Sample wafer rotation angle
      std::uniform_real_distribution<NumericType> uniform;
      NumericType phi_stage = 2. * M_PI * uniform(rngState);

      // 2) Beam axis a (tilted by alpha from -z)
      NumericType sin_phi_stage = std::sin(phi_stage),
                  cos_phi_stage = std::cos(phi_stage);
      NumericType sin_alpha = std::sin(tiltAngle_),
                  cos_alpha = std::cos(tiltAngle_);
      Vec3D<NumericType> a{sin_alpha * cos_phi_stage, sin_alpha * sin_phi_stage,
                           -cos_alpha}; // already unit length

      // 3) Build basis (e1, e2, e3)
      auto e3 = a;
      auto h = (abs(e3[2]) < 0.9) ? Vec3D<NumericType>{0, 0, 1}
                                  : Vec3D<NumericType>{1, 0, 0};
      auto e1 = h - e3 * DotProduct(h, e3);
      Normalize(e1);
      auto e2 = CrossProduct(e3, e1);

      // 4) Sample power-cosine around e3
      NumericType cosTheta =
          std::pow(uniform(rngState), 1. / (params_.exponent + 1.));
      NumericType sinTheta = std::sqrt(std::max(0., 1. - cosTheta * cosTheta));
      NumericType phi = 2. * M_PI * uniform(rngState);

      NumericType lx = sinTheta * std::cos(phi);
      NumericType ly = sinTheta * std::sin(phi);
      NumericType lz = cosTheta;

      auto direction = lx * e1 + ly * e2 + lz * e3;

      // 5) ensure downward (toward wafer)
      if (direction[2] >= 0.f) {
        direction[0] = -direction[0];
        direction[1] = -direction[1];
        direction[2] = -direction[2];
      }

      if constexpr (D == 2) {
        direction[1] = direction[2];
        direction[2] = 0.f;
        Normalize(direction);
      }

      return direction;
    } else {
      return Vec3D<NumericType>{0, 0, 0};
    }
  }

  NumericType getSourceDistributionPower() const override {
    return params_.exponent;
  }

  std::vector<std::string> getLocalDataLabels() const override {
    return {IBESurfaceModel<NumericType>::fluxLabel,
            IBESurfaceModel<NumericType>::redepositionLabel};
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
  const NumericType tiltAngle_; // in rad
};
} // namespace impl

#ifdef VIENNACORE_COMPILE_GPU
namespace gpu {

template <typename NumericType, int D>
class IonBeamEtching : public ProcessModelGPU<NumericType, D> {
public:
  IonBeamEtching(const IBEParameters<NumericType> &params,
                 const std::vector<Material> &maskMaterial)
      : maskMaterials_(maskMaterial), params_(params) {
    // particles
    viennaray::gpu::Particle<NumericType> particle{
        .name = "IBEIon", .cosineExponent = params_.exponent};
    particle.dataLabels.push_back(
        ::viennaps::impl::IBESurfaceModel<NumericType>::fluxLabel);
    if (params_.redepositionRate > 0.) {
      particle.dataLabels.push_back(
          ::viennaps::impl::IBESurfaceModel<NumericType>::redepositionLabel);
    }

    if (params_.tiltAngle != 0. && !params_.rotatingWafer) {
      Vec3D<NumericType> direction{0., 0., 0.};
      direction[0] = std::sin(constants::degToRad(params_.tiltAngle));
      direction[D - 1] = -std::cos(constants::degToRad(params_.tiltAngle));
      particle.direction = direction;
      particle.useCustomDirection = true;
    }

    // Callables
    std::unordered_map<std::string, unsigned> pMap = {{"IBEIon", 0}};
    std::vector<viennaray::gpu::CallableConfig> cMap = {
        {0, viennaray::gpu::CallableSlot::COLLISION,
         "__direct_callable__IBECollision"},
        {0, viennaray::gpu::CallableSlot::REFLECTION,
         "__direct_callable__IBEReflection"},
        {0, viennaray::gpu::CallableSlot::INIT, "__direct_callable__IBEInit"}};
    this->setParticleCallableMap(pMap, cMap);

    // Parameters to upload to device
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
    deviceParams.redepositionRate =
        static_cast<float>(params_.redepositionRate);
    deviceParams.redepositionThreshold =
        static_cast<float>(params_.redepositionThreshold);

    deviceParams.tiltAngle =
        static_cast<float>(constants::degToRad(params_.tiltAngle));
    deviceParams.rotating = params_.rotatingWafer;

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
  }

  ~IonBeamEtching() override { this->processData.free(); }

private:
  std::vector<Material> maskMaterials_;
  IBEParameters<NumericType> params_;
};

} // namespace gpu
#endif

template <typename NumericType, int D>
class IonBeamEtching : public ProcessModelCPU<NumericType, D> {
public:
  IonBeamEtching(const IBEParameters<NumericType> &params)
      : IonBeamEtching(params, {}) {}

  IonBeamEtching(const IBEParameters<NumericType> &params,
                 const std::vector<Material> &maskMaterial)
      : maskMaterials_(maskMaterial), params_(params) {

    // particles
    auto particle =
        std::make_unique<impl::IBEIonWithRedeposition<NumericType, D>>(params_);

    // surface model
    auto surfModel = SmartPointer<impl::IBESurfaceModel<NumericType>>::New(
        params_, maskMaterials_);

    // velocity field
    auto velField = SmartPointer<DefaultVelocityField<NumericType, D>>::New();

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->insertNextParticleType(particle);
    this->setProcessName("IonBeamEtching");
    this->processMetaData = params_.toProcessMetaData();
    this->hasGPU = true;
  }

#ifdef VIENNACORE_COMPILE_GPU
  SmartPointer<ProcessModelBase<NumericType, D>> getGPUModel() final {
    auto model = SmartPointer<gpu::IonBeamEtching<NumericType, D>>::New(
        params_, maskMaterials_);
    model->setProcessName(this->getProcessName().value());
    return model;
  }
#endif

private:
  std::vector<Material> maskMaterials_;
  IBEParameters<NumericType> params_;
};

PS_PRECOMPILE_PRECISION_DIMENSION(IonBeamEtching)

} // namespace viennaps
