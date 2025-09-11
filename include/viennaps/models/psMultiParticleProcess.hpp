#pragma once

#include "../process/psProcessModel.hpp"
#include "../psMaterials.hpp"

#include <rayParticle.hpp>
#include <rayReflection.hpp>

#include <numeric>

namespace viennaps {

using namespace viennacore;

namespace impl {
template <typename NumericType, int D>
class MultiParticleSurfaceModel : public viennaps::SurfaceModel<NumericType> {
public:
  std::function<NumericType(const std::vector<NumericType> &, const Material &)>
      rateFunction_;
  std::vector<std::string> &fluxDataLabels_;

public:
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

    auto cosTheta = std::clamp(-DotProduct(rayDir, geomNormal), NumericType(0),
                               NumericType(1));
    NumericType incomingAngle = std::acos(cosTheta);
    assert(incomingAngle <= M_PI_2 + 1e-6 && "Error in calculating angle");
    assert(incomingAngle >= 0 && "Error in calculating angle");

    if (energy_ > 0.) {
      // Small incident angles are reflected with the energy fraction centered
      // at 0
      NumericType Eref_peak;
      if (incomingAngle >= inflectAngle_) {
        Eref_peak = (1 - (1 - A_) * (M_PI_2 - incomingAngle) /
                             (M_PI_2 - inflectAngle_));
      } else {
        Eref_peak = A_ * std::pow(incomingAngle / inflectAngle_, n_);
      }
      // Gaussian distribution around the Eref_peak scaled by the particle
      // energy
      NumericType newEnergy;
      std::normal_distribution<NumericType> normalDist(energy_ * Eref_peak,
                                                       0.1 * energy_);
      do {
        newEnergy = normalDist(rngState);
      } while (newEnergy > energy_ || newEnergy < 0.);
      energy_ = newEnergy;
    }

    NumericType sticking = 1.;
    if (incomingAngle > thetaRMin_)
      sticking = 1. - std::min((incomingAngle - thetaRMin_) /
                                   (thetaRMax_ - thetaRMin_),
                               NumericType(1.));

    auto direction = viennaray::ReflectionConedCosine<NumericType, D>(
        rayDir, geomNormal, rngState,
        M_PI_2 - std::min(incomingAngle, minAngle_));

    return std::pair<NumericType, Vec3D<NumericType>>{sticking, direction};
  }
  void initNew(RNG &rngState) override final {
    energy_ = -1.;
    if (meanEnergy_ > 0.) {
      std::normal_distribution<NumericType> normalDist{meanEnergy_,
                                                       sigmaEnergy_};
      do {
        energy_ = normalDist(rngState);
      } while (energy_ <= 0.);
    }
  }
  NumericType getSourceDistributionPower() const override final {
    return sourcePower_;
  }
  std::vector<std::string> getLocalDataLabels() const override final {
    return {dataLabel_};
  }

private:
  NumericType energy_;

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
  std::normal_distribution<NumericType> normalDist_;

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

template <typename NumericType, int D>
class MultiParticleProcess : public ProcessModelCPU<NumericType, D> {
public:
  MultiParticleProcess() {
    // surface model
    auto surfModel =
        SmartPointer<impl::MultiParticleSurfaceModel<NumericType, D>>::New(
            fluxDataLabels_);

    // velocity field
    auto velField = SmartPointer<DefaultVelocityField<NumericType, D>>::New(2);

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
        thetaRMin * M_PI / 180., thetaRMax * M_PI / 180.,
        inflectAngle * M_PI / 180., minAngle * M_PI / 180., n, dataLabel);
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

private:
  std::vector<std::string> fluxDataLabels_;
  using ProcessModelCPU<NumericType, D>::processMetaData;

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
