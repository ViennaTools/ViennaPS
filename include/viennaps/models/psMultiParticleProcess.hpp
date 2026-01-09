#pragma once

#include "../process/psProcessModel.hpp"
#include "../psConstants.hpp"
#include "../psMaterials.hpp"
#include "../psUtil.hpp"
#include "psIonModelUtil.hpp"
#include "psPipelineParameters.hpp"

#include <rayParticle.hpp>
#include <rayReflection.hpp>

#ifdef VIENNACORE_COMPILE_GPU
#include <gpu/raygCallableConfig.hpp>
#endif

#include <numeric>

namespace viennaps {

using namespace viennacore;

namespace impl {
template <typename NumericType, int D>
class MultiParticleSurfaceModel : public SurfaceModel<NumericType> {
public:
  std::function<NumericType(const std::vector<NumericType> &, const Material &)>
      rateFunction_;
  std::vector<std::string> &fluxDataLabels_;

  MultiParticleSurfaceModel(std::vector<std::string> &fluxDataLabels)
      : fluxDataLabels_(fluxDataLabels) {
    rateFunction_ = [](const std::vector<NumericType> &fluxes,
                       const Material &material) {
      return std::accumulate(fluxes.begin(), fluxes.end(), 0.);
    };
  }

  SmartPointer<std::vector<NumericType>>
  calculateVelocities(SmartPointer<viennals::PointData<NumericType>> rates,
                      const std::vector<Vec3D<NumericType>> &coordinates,
                      const std::vector<NumericType> &materialIds) override {

    auto velocity =
        SmartPointer<std::vector<NumericType>>::New(materialIds.size(), 0.);

    std::vector<std::vector<NumericType> *> fluxPtrs;
    for (const auto &label : fluxDataLabels_) {
      fluxPtrs.push_back(rates->getScalarData(label));
    }

    std::vector<NumericType> fluxes(fluxPtrs.size());
    for (std::size_t i = 0; i < velocity->size(); i++) {
      for (std::size_t j = 0; j < fluxPtrs.size(); j++) {
        fluxes[j] = fluxPtrs[j]->at(i);
      }
      velocity->at(i) =
          rateFunction_(fluxes, MaterialMap::mapToMaterial(materialIds[i]));
    }

    return velocity;
  }
};

template <typename NumericType, int D>
class IonParticle
    : public viennaray::Particle<IonParticle<NumericType, D>, NumericType> {
public:
  IonParticle(NumericType sourcePower, NumericType meanEnergy,
              NumericType sigmaEnergy, NumericType thresholdEnergy,
              NumericType B_sp, NumericType thetaRMin, NumericType thetaRMax,
              NumericType inflectAngle, NumericType minAngle, NumericType n,
              const std::string &dataLabel)
      : sourcePower_(sourcePower), meanEnergy_(meanEnergy),
        sigmaEnergy_(sigmaEnergy), thresholdEnergy_(thresholdEnergy),
        B_sp_(B_sp), thetaRMin_(thetaRMin), thetaRMax_(thetaRMax),
        inflectAngle_(inflectAngle), minAngle_(minAngle),
        A_(1. / (1. + n * (M_PI_2 / inflectAngle - 1.))), n_(n),
        dataLabel_(dataLabel) {}

  void surfaceCollision(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                        const Vec3D<NumericType> &geomNormal,
                        const unsigned int primID, const int,
                        viennaray::TracingData<NumericType> &localData,
                        const viennaray::TracingData<NumericType> *,
                        RNG &) override final {
    NumericType flux = rayWeight;

    if (B_sp_ >= 0.) {
      NumericType cosTheta = -DotProduct(rayDir, geomNormal);
      // if (cosTheta < 0.5)
      // flux *= std::max(3. - 6. * angle / M_PI, 0.);
      flux *= (1 + B_sp_ * (1 - cosTheta * cosTheta)) * cosTheta;
    }

    if (energy_ > 0.) {
      flux *= std::max(std::sqrt(energy_) - std::sqrt(thresholdEnergy_),
                       NumericType(0.));
    }

    localData.getVectorData(0)[primID] += flux;
  }
  std::pair<NumericType, Vec3D<NumericType>>
  surfaceReflection(NumericType, const Vec3D<NumericType> &rayDir,
                    const Vec3D<NumericType> &geomNormal, const unsigned int,
                    const int, const viennaray::TracingData<NumericType> *,
                    RNG &rngState) override final {

    auto cosTheta = getCosTheta(rayDir, geomNormal);
    NumericType incomingAngle = std::acos(cosTheta);

    NumericType sticking = 1.;
    if (incomingAngle > thetaRMin_) {
      sticking = 1. - util::saturate((incomingAngle - thetaRMin_) /
                                     (thetaRMax_ - thetaRMin_));
    }

    if (sticking >= 1.) {
      return VIENNARAY_PARTICLE_STOP;
    }

    if (energy_ > 0.) {
      energy_ =
          updateEnergy(rngState, energy_, incomingAngle, A_, inflectAngle_, n_);
    }

    auto direction = viennaray::ReflectionConedCosine<NumericType, D>(
        rayDir, geomNormal, rngState,
        M_PI_2 - std::min(incomingAngle, minAngle_));

    return std::pair<NumericType, Vec3D<NumericType>>{sticking, direction};
  }
  void initNew(RNG &rngState) override final {
    if (meanEnergy_ > 0.) {
      energy_ = initNormalDistEnergy(rngState, meanEnergy_, sigmaEnergy_);
    }
  }
  NumericType getSourceDistributionPower() const override final {
    return sourcePower_;
  }
  std::vector<std::string> getLocalDataLabels() const override final {
    return {dataLabel_};
  }

private:
  NumericType energy_ = 0.;

  const NumericType sourcePower_;

  const NumericType meanEnergy_;
  const NumericType sigmaEnergy_;
  const NumericType thresholdEnergy_;

  const NumericType B_sp_;

  const NumericType thetaRMin_;
  const NumericType thetaRMax_;

  const NumericType inflectAngle_;
  const NumericType minAngle_;
  const NumericType A_;
  const NumericType n_;

  const std::string dataLabel_;
};

template <typename NumericType, int D>
class DiffuseParticle
    : public viennaray::Particle<DiffuseParticle<NumericType, D>, NumericType> {
public:
  DiffuseParticle(NumericType stickingProbability, const std::string &dataLabel)
      : stickingProbability_(stickingProbability), dataLabel_(dataLabel) {}

  DiffuseParticle(NumericType stickingProbability,
                  std::unordered_map<Material, NumericType> materialSticking,
                  const std::string &dataLabel)
      : materialSticking_(materialSticking),
        stickingProbability_(stickingProbability), dataLabel_(dataLabel) {}

  void surfaceCollision(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                        const Vec3D<NumericType> &geomNormal,
                        const unsigned int primID, const int,
                        viennaray::TracingData<NumericType> &localData,
                        const viennaray::TracingData<NumericType> *,
                        RNG &) override final {
    localData.getVectorData(0)[primID] += rayWeight;
  }
  std::pair<NumericType, Vec3D<NumericType>>
  surfaceReflection(NumericType, const Vec3D<NumericType> &rayDir,
                    const Vec3D<NumericType> &geomNormal, const unsigned int,
                    const int materialId,
                    const viennaray::TracingData<NumericType> *,
                    RNG &rngState) override final {
    NumericType sticking = stickingProbability_;
    if (auto mat =
            materialSticking_.find(MaterialMap::mapToMaterial(materialId));
        mat != materialSticking_.end()) {
      sticking = mat->second;
    }

    auto direction =
        viennaray::ReflectionDiffuse<NumericType, D>(geomNormal, rngState);
    return std::pair<NumericType, Vec3D<NumericType>>{sticking, direction};
  }
  std::vector<std::string> getLocalDataLabels() const override final {
    return {dataLabel_};
  }

private:
  const std::unordered_map<Material, NumericType> materialSticking_;
  const NumericType stickingProbability_;
  const std::string dataLabel_;
};
} // namespace impl

#ifdef VIENNACORE_COMPILE_GPU
namespace gpu {

template <typename NumericType, int D>
class MultiParticleProcess final : public ProcessModelGPU<NumericType, D> {
public:
  MultiParticleProcess() {
    // surface model
    auto surfModel = SmartPointer<::viennaps::impl::MultiParticleSurfaceModel<
        NumericType, D>>::New(fluxDataLabels_);

    // velocity field
    auto velField = SmartPointer<DefaultVelocityField<NumericType, D>>::New();

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("MultiParticleProcess");
    this->hasGPU = true;

    // Callables
    std::unordered_map<std::string, unsigned> pMap = {{"Neutral", 0},
                                                      {"Ion", 1}};
    std::vector<viennaray::gpu::CallableConfig> cMap = {
        {0, viennaray::gpu::CallableSlot::COLLISION,
         "__direct_callable__multiNeutralCollision"},
        {0, viennaray::gpu::CallableSlot::REFLECTION,
         "__direct_callable__multiNeutralReflection"},
        {1, viennaray::gpu::CallableSlot::COLLISION,
         "__direct_callable__multiIonCollision"},
        {1, viennaray::gpu::CallableSlot::REFLECTION,
         "__direct_callable__multiIonReflection"},
        {1, viennaray::gpu::CallableSlot::INIT,
         "__direct_callable__multiIonInit"}};
    this->setParticleCallableMap(pMap, cMap);
  }

  MultiParticleProcess(
      const std::vector<std::string> &fluxDataLabels,
      std::unordered_map<std::string, NumericType> const &ionParams,
      std::unordered_map<std::string, NumericType> const &neutralParams,
      std::function<NumericType(const std::vector<NumericType> &,
                                const Material &)>
          rateFunction) {
    // surface model
    auto surfModel = SmartPointer<::viennaps::impl::MultiParticleSurfaceModel<
        NumericType, D>>::New(fluxDataLabels_);
    surfModel->rateFunction_ = rateFunction;

    // velocity field
    auto velField = SmartPointer<DefaultVelocityField<NumericType, D>>::New();

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("MultiParticleProcess");
    this->hasGPU = true;

    // Callables
    std::unordered_map<std::string, unsigned> pMap = {{"Neutral", 0},
                                                      {"Ion", 1}};
    std::vector<viennaray::gpu::CallableConfig> cMap = {
        {0, viennaray::gpu::CallableSlot::COLLISION,
         "__direct_callable__multiNeutralCollision"},
        {0, viennaray::gpu::CallableSlot::REFLECTION,
         "__direct_callable__multiNeutralReflection"},
        {1, viennaray::gpu::CallableSlot::COLLISION,
         "__direct_callable__multiIonCollision"},
        {1, viennaray::gpu::CallableSlot::REFLECTION,
         "__direct_callable__multiIonReflection"},
        {1, viennaray::gpu::CallableSlot::INIT,
         "__direct_callable__multiIonInit"}};
    this->setParticleCallableMap(pMap, cMap);

    for (auto const &label : fluxDataLabels) {
      if (label.find("ion") != std::string::npos) {
        assert(ionParams.size() > 0 &&
               "Ion parameters must be provided when adding ion flux.");
        addIonParticle(ionParams.at("SourcePower"), ionParams.at("ThetaRMin"),
                       ionParams.at("ThetaRMax"), ionParams.at("MinAngle"),
                       ionParams.at("B_sp"), ionParams.at("MeanEnergy"),
                       ionParams.at("SigmaEnergy"),
                       ionParams.at("ThresholdEnergy"),
                       ionParams.at("InflectAngle"), ionParams.at("n"), label);
      } else if (label.find("neutral") != std::string::npos) {
        assert(neutralParams.size() > 0 &&
               "Neutral parameters must be provided when adding neutral flux.");
        addNeutralParticle(neutralParams.at("StickingProbability"), label);
      }
    }
  }

  void addNeutralParticle(NumericType stickingProbability,
                          const std::string &label = "neutralFlux") {
    std::string dataLabel = label + std::to_string(fluxDataLabels_.size());
    fluxDataLabels_.push_back(dataLabel);
    viennaray::gpu::Particle<NumericType> particle;
    particle.name = "Neutral";
    particle.sticking = stickingProbability;
    particle.dataLabels.push_back(dataLabel);
    particle.materialSticking[static_cast<int>(Material::Undefined)] =
        1.; // this will initialize all to default sticking

    this->insertNextParticleType(particle);
    this->setUseMaterialIds(true);

    addStickingData(stickingProbability);
  }

  void
  addNeutralParticle(std::unordered_map<Material, NumericType> materialSticking,
                     NumericType defaultStickingProbability = 1.,
                     const std::string &label = "neutralFlux") {
    std::string dataLabel = label + std::to_string(fluxDataLabels_.size());
    fluxDataLabels_.push_back(dataLabel);

    viennaray::gpu::Particle<NumericType> particle;
    particle.name = "Neutral";
    particle.sticking = defaultStickingProbability;
    particle.dataLabels.push_back(dataLabel);
    for (auto &mat : materialSticking) {
      particle.materialSticking[static_cast<int>(mat.first)] = mat.second;
    }

    this->insertNextParticleType(particle);
    this->setUseMaterialIds(true);

    addStickingData(defaultStickingProbability);
  }

  void addIonParticle(NumericType sourcePower, NumericType thetaRMin = 0.,
                      NumericType thetaRMax = 90., NumericType minAngle = 80.,
                      NumericType B_sp = -1., NumericType meanEnergy = 0.,
                      NumericType sigmaEnergy = 0.,
                      NumericType thresholdEnergy = 0.,
                      NumericType inflectAngle = 0., NumericType n = 1,
                      const std::string &label = "ionFlux") {
    if (ionCount_ > 0) {
      VIENNACORE_LOG_WARNING("GPU MultiParticleProcess currently only "
                             "supports one ion particle type.");
      return;
    }

    std::string dataLabel = label + std::to_string(fluxDataLabels_.size());
    fluxDataLabels_.push_back(dataLabel);

    viennaray::gpu::Particle<NumericType> particle;
    particle.name = "Ion";
    particle.dataLabels.push_back(dataLabel);
    setDirection(particle);
    particle.cosineExponent = sourcePower;

    impl::IonParams params;
    params.thetaRMin = constants::degToRad(thetaRMin);
    params.thetaRMax = constants::degToRad(thetaRMax);
    params.minAngle = constants::degToRad(minAngle);
    params.B_sp = B_sp;
    params.meanEnergy = meanEnergy;
    params.sigmaEnergy = sigmaEnergy;
    params.thresholdEnergy = std::sqrt(thresholdEnergy);
    params.inflectAngle = constants::degToRad(inflectAngle);
    params.n_l = n;
    this->processData.allocUploadSingle(params);
    this->insertNextParticleType(particle);

    addIonData({{"SourcePower", sourcePower},
                {"MeanEnergy", meanEnergy},
                {"SigmaEnergy", sigmaEnergy},
                {"ThresholdEnergy", thresholdEnergy},
                {"B_sp", B_sp},
                {"ThetaRMin", thetaRMin},
                {"ThetaRMax", thetaRMax},
                {"InflectAngle", inflectAngle},
                {"MinAngle", minAngle},
                {"n", n}});
  }

  void
  setRateFunction(std::function<NumericType(const std::vector<NumericType> &,
                                            const Material &)>
                      rateFunction) {
    auto surfModel = std::dynamic_pointer_cast<
        viennaps::impl::MultiParticleSurfaceModel<NumericType, D>>(
        this->getSurfaceModel());
    surfModel->rateFunction_ = rateFunction;
  }

  ~MultiParticleProcess() { this->processData.free(); }

private:
  std::vector<std::string> fluxDataLabels_;
  using ProcessModelBase<NumericType, D>::processMetaData;
  unsigned int ionCount_ = 0;

  void setDirection(viennaray::gpu::Particle<NumericType> &particle) {
    auto direction = this->getPrimaryDirection();
    if (direction.has_value()) {
      VIENNACORE_LOG_DEBUG("Using custom direction for ion particle: " +
                           util::arrayToString(direction.value()));
      particle.direction = direction.value();
      particle.useCustomDirection = true;
    }
  }

  void addStickingData(NumericType stickingProbability) {
    if (processMetaData.find("StickingProbability") == processMetaData.end()) {
      processMetaData["StickingProbability"] =
          std::vector<double>{stickingProbability};
    } else {
      processMetaData["StickingProbability"].push_back(stickingProbability);
    }
  }

  void addIonData(std::vector<std::pair<std::string, NumericType>> data) {
    for (const auto &pair : data) {
      if (processMetaData.find(pair.first) == processMetaData.end()) {
        processMetaData[pair.first] = std::vector<double>{pair.second};
      } else {
        processMetaData[pair.first].push_back(pair.second);
      }
    }
    ++ionCount_;
  }
};

} // namespace gpu
#endif

template <typename NumericType, int D>
class MultiParticleProcess : public ProcessModelCPU<NumericType, D> {
public:
  MultiParticleProcess() {
    // surface model
    auto surfModel =
        SmartPointer<impl::MultiParticleSurfaceModel<NumericType, D>>::New(
            fluxDataLabels_);

    // velocity field
    auto velField = SmartPointer<DefaultVelocityField<NumericType, D>>::New();

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("MultiParticleProcess");
    this->hasGPU = true;
  }

  void addNeutralParticle(NumericType stickingProbability,
                          const std::string &label = "neutralFlux") {
    std::string dataLabel = label + std::to_string(fluxDataLabels_.size());
    fluxDataLabels_.push_back(dataLabel);
    auto particle = std::make_unique<impl::DiffuseParticle<NumericType, D>>(
        stickingProbability, dataLabel);
    this->insertNextParticleType(particle);

    addStickingData(stickingProbability);
  }

  void
  addNeutralParticle(std::unordered_map<Material, NumericType> materialSticking,
                     NumericType defaultStickingProbability = 1.,
                     const std::string &label = "neutralFlux") {
    std::string dataLabel = label + std::to_string(fluxDataLabels_.size());
    fluxDataLabels_.push_back(dataLabel);
    auto particle = std::make_unique<impl::DiffuseParticle<NumericType, D>>(
        defaultStickingProbability, materialSticking, dataLabel);
    this->insertNextParticleType(particle);

    addStickingData(defaultStickingProbability);
  }

  void addIonParticle(NumericType sourcePower, NumericType thetaRMin = 0.,
                      NumericType thetaRMax = 90., NumericType minAngle = 80.,
                      NumericType B_sp = -1., NumericType meanEnergy = 0.,
                      NumericType sigmaEnergy = 0.,
                      NumericType thresholdEnergy = 0.,
                      NumericType inflectAngle = 0., NumericType n = 1,
                      const std::string &label = "ionFlux") {
    std::string dataLabel = label + std::to_string(fluxDataLabels_.size());
    fluxDataLabels_.push_back(dataLabel);
    auto particle = std::make_unique<impl::IonParticle<NumericType, D>>(
        sourcePower, meanEnergy, sigmaEnergy, thresholdEnergy, B_sp,
        constants::degToRad(thetaRMin), constants::degToRad(thetaRMax),
        constants::degToRad(inflectAngle), constants::degToRad(minAngle), n,
        dataLabel);
    this->insertNextParticleType(particle);

    addIonData({{"SourcePower", sourcePower},
                {"MeanEnergy", meanEnergy},
                {"SigmaEnergy", sigmaEnergy},
                {"ThresholdEnergy", thresholdEnergy},
                {"B_sp", B_sp},
                {"ThetaRMin", thetaRMin},
                {"ThetaRMax", thetaRMax},
                {"InflectAngle", inflectAngle},
                {"MinAngle", minAngle},
                {"n", n}});
  }

  void
  setRateFunction(std::function<NumericType(const std::vector<NumericType> &,
                                            const Material &)>
                      rateFunction) {
    auto surfModel = std::dynamic_pointer_cast<
        impl::MultiParticleSurfaceModel<NumericType, D>>(
        this->getSurfaceModel());
    surfModel->rateFunction_ = rateFunction;
  }

#ifdef VIENNACORE_COMPILE_GPU
  SmartPointer<ProcessModelBase<NumericType, D>> getGPUModel() override {
    if (ionParams_.empty() && neutralParams_.empty()) {
      VIENNACORE_LOG_WARNING(
          "Cannot convert MultiParticleProcess to GPU model without any "
          "particles added.");
      return nullptr;
    }
    auto surfModel = std::dynamic_pointer_cast<
        impl::MultiParticleSurfaceModel<NumericType, D>>(
        this->getSurfaceModel());
    auto model = SmartPointer<gpu::MultiParticleProcess<NumericType, D>>::New(
        fluxDataLabels_, ionParams_, neutralParams_, surfModel->rateFunction_);
    model->setProcessName(this->getProcessName().value());
    return model;
  }
#endif

private:
  std::vector<std::string> fluxDataLabels_;
  using ProcessModelCPU<NumericType, D>::processMetaData;
  std::unordered_map<std::string, NumericType> ionParams_;
  std::unordered_map<std::string, NumericType> neutralParams_;

  void addStickingData(NumericType stickingProbability) {
    if (processMetaData.find("StickingProbability") == processMetaData.end()) {
      addMetaData("StickingProbability", stickingProbability);
    } else {
      processMetaData["StickingProbability"].push_back(stickingProbability);
    }
    neutralParams_["StickingProbability"] = stickingProbability;
  }

  void addIonData(std::vector<std::pair<std::string, NumericType>> data) {
    for (const auto &pair : data) {
      if (processMetaData.find(pair.first) == processMetaData.end()) {
        addMetaData(pair.first + " Rate", pair.second);
      } else {
        processMetaData[pair.first].push_back(pair.second);
      }
      ionParams_[pair.first] = pair.second;
    }
  }

  inline void addMetaData(const std::string &key, double value) {
    processMetaData[key] = std::vector<double>{value};
  }
};

PS_PRECOMPILE_PRECISION_DIMENSION(MultiParticleProcess)

} // namespace viennaps
