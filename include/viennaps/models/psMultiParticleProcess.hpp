#pragma once

#include "../materials/psMaterials.hpp"
#include "../process/psProcessModel.hpp"
#include "../psConstants.hpp"
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
        sqrtThresholdEnergy_(std::sqrt(thresholdEnergy)), B_sp_(B_sp),
        thetaRMin_(thetaRMin), thetaRMax_(thetaRMax),
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
      flux *=
          std::max(std::sqrt(energy_) - sqrtThresholdEnergy_, NumericType(0.));
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

public:
  NumericType energy_ = 0.;

  const NumericType sourcePower_;

  const NumericType meanEnergy_;
  const NumericType sigmaEnergy_;
  const NumericType thresholdEnergy_;
  const NumericType sqrtThresholdEnergy_;

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

public:
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
      std::vector<::viennaps::impl::IonParticle<NumericType, D>> const &ions,
      std::vector<::viennaps::impl::DiffuseParticle<NumericType, D>> const
          &neutrals,
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

    if (ions.size() > 1) {
      VIENNACORE_LOG_WARNING(
          "GPU MultiParticleProcess currently only supports one ion particle "
          "type. Additional ion particles will be ignored.");
    }
    bool ionAdded = false;
    std::vector<bool> neutralAdded(neutrals.size(), false);

    for (auto const &label : fluxDataLabels) {
      if (!ions.empty() && label == ions.front().dataLabel_) {
        const auto &ion = ions.front();
        addIonParticle(ion.sourcePower_, ion.thetaRMin_, ion.thetaRMax_,
                       ion.minAngle_, ion.B_sp_, ion.meanEnergy_,
                       ion.sigmaEnergy_, ion.thresholdEnergy_,
                       ion.inflectAngle_, ion.n_, ion.dataLabel_);
        ionAdded = true;
      }

      for (std::size_t i = 0; i < neutrals.size(); ++i) {
        if (!neutralAdded[i] && label == neutrals[i].dataLabel_) {
          const auto &neutral = neutrals[i];
          if (neutral.materialSticking_.empty()) {
            addNeutralParticle(neutral.stickingProbability_,
                               neutral.dataLabel_);
          } else {
            addNeutralParticle(neutral.materialSticking_,
                               neutral.stickingProbability_,
                               neutral.dataLabel_);
          }
          neutralAdded[i] = true;
          break;
        }
      }
    }

    if (!ions.empty() && !ionAdded) {
      const auto &ion = ions.front();
      VIENNACORE_LOG_WARNING(
          "Ion label was not found in flux label ordering during CPU->GPU "
          "conversion. Falling back to first ion particle.");
      addIonParticle(ion.sourcePower_, ion.thetaRMin_, ion.thetaRMax_,
                     ion.minAngle_, ion.B_sp_, ion.meanEnergy_,
                     ion.sigmaEnergy_, ion.thresholdEnergy_, ion.inflectAngle_,
                     ion.n_, ion.dataLabel_);
    }

    for (std::size_t i = 0; i < neutrals.size(); ++i) {
      if (neutralAdded[i]) {
        continue;
      }

      const auto &neutral = neutrals[i];
      VIENNACORE_LOG_WARNING(
          "Neutral label was not found in flux label ordering during CPU->GPU "
          "conversion. Appending neutral particle with label: " +
          neutral.dataLabel_);
      if (neutral.materialSticking_.empty()) {
        addNeutralParticle(neutral.stickingProbability_, neutral.dataLabel_);
      } else {
        addNeutralParticle(neutral.materialSticking_,
                           neutral.stickingProbability_, neutral.dataLabel_);
      }
    }

    VIENNACORE_LOG_DEBUG("GPU MultiParticleProcess conversion completed with " +
                         std::to_string(ionCount_) + " ion particle(s) and " +
                         std::to_string(neutralCount_) +
                         " neutral particle(s).");
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
    ++neutralCount_;
    VIENNACORE_LOG_DEBUG("Added neutral particle with sticking probability: " +
                         std::to_string(stickingProbability));
  }

  void addNeutralParticle(
      const std::unordered_map<Material, NumericType> &materialSticking,
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
    ++neutralCount_;
    VIENNACORE_LOG_DEBUG(
        "Added neutral particle with default sticking probability: " +
        std::to_string(defaultStickingProbability) +
        " and material-specific sticking.");
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
    VIENNACORE_LOG_DEBUG(
        "Added ion particle with source power: " + std::to_string(sourcePower) +
        ", thetaRMin: " + std::to_string(thetaRMin) + ", thetaRMax: " +
        std::to_string(thetaRMax) + ", minAngle: " + std::to_string(minAngle) +
        ", B_sp: " + std::to_string(B_sp) +
        ", meanEnergy: " + std::to_string(meanEnergy) +
        ", sigmaEnergy: " + std::to_string(sigmaEnergy) +
        ", thresholdEnergy: " + std::to_string(thresholdEnergy) +
        ", inflectAngle: " + std::to_string(inflectAngle) +
        ", n: " + std::to_string(n));
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
  unsigned int neutralCount_ = 0;

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
    neutralParticles_.emplace_back(stickingProbability, dataLabel);
  }

  void addNeutralParticle(
      const std::unordered_map<Material, NumericType> &materialSticking,
      NumericType defaultStickingProbability = 1.,
      const std::string &label = "neutralFlux") {
    std::string dataLabel = label + std::to_string(fluxDataLabels_.size());
    fluxDataLabels_.push_back(dataLabel);
    auto particle = std::make_unique<impl::DiffuseParticle<NumericType, D>>(
        defaultStickingProbability, materialSticking, dataLabel);
    this->insertNextParticleType(particle);

    addStickingData(defaultStickingProbability);
    neutralParticles_.emplace_back(defaultStickingProbability, materialSticking,
                                   dataLabel);
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
    ionParticles_.emplace_back(sourcePower, meanEnergy, sigmaEnergy,
                               thresholdEnergy, B_sp, thetaRMin, thetaRMax,
                               inflectAngle, minAngle, n, dataLabel);
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
    if (ionParticles_.empty() && neutralParticles_.empty()) {
      VIENNACORE_LOG_WARNING(
          "Cannot convert MultiParticleProcess to GPU model without any "
          "particles added.");
      return nullptr;
    }
    if (!ionParticles_.empty() && ionParticles_.size() > 1) {
      VIENNACORE_LOG_WARNING(
          "GPU MultiParticleProcess currently only supports one ion particle "
          "type. Only the first ion particle will be converted.");
    }

    auto surfModel = std::dynamic_pointer_cast<
        impl::MultiParticleSurfaceModel<NumericType, D>>(
        this->getSurfaceModel());
    if (!surfModel) {
      VIENNACORE_LOG_WARNING(
          "Failed to access MultiParticleSurfaceModel during CPU->GPU "
          "conversion.");
      return nullptr;
    }

    VIENNACORE_LOG_DEBUG(
        "Converting MultiParticleProcess to GPU with " +
        std::to_string(std::min<std::size_t>(ionParticles_.size(), 1)) +
        " ion and " + std::to_string(neutralParticles_.size()) +
        " neutral particle(s); flux labels: " +
        std::to_string(fluxDataLabels_.size()));

    auto model = SmartPointer<gpu::MultiParticleProcess<NumericType, D>>::New(
        fluxDataLabels_, ionParticles_, neutralParticles_,
        surfModel->rateFunction_);
    model->setProcessName(this->getProcessName().value());
    return model;
  }
#endif

private:
  std::vector<std::string> fluxDataLabels_;
  using ProcessModelCPU<NumericType, D>::processMetaData;
  std::vector<impl::IonParticle<NumericType, D>> ionParticles_;
  std::vector<impl::DiffuseParticle<NumericType, D>> neutralParticles_;

  void addStickingData(NumericType stickingProbability) {
    if (processMetaData.find("StickingProbability") == processMetaData.end()) {
      addMetaData("StickingProbability", stickingProbability);
    } else {
      processMetaData["StickingProbability"].push_back(stickingProbability);
    }
  }

  void addIonData(std::vector<std::pair<std::string, NumericType>> data) {
    for (const auto &pair : data) {
      if (processMetaData.find(pair.first) == processMetaData.end()) {
        addMetaData(pair.first + " Rate", pair.second);
      } else {
        processMetaData[pair.first].push_back(pair.second);
      }
    }
  }

  inline void addMetaData(const std::string &key, double value) {
    processMetaData[key] = std::vector<double>{value};
  }
};

PS_PRECOMPILE_PRECISION_DIMENSION(MultiParticleProcess)

} // namespace viennaps
