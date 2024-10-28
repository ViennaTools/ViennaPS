#pragma once

#include "../psMaterials.hpp"
#include "../psProcessModel.hpp"

#include <rayParticle.hpp>
#include <rayReflection.hpp>

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
      : fluxDataLabels_(fluxDataLabels) {}

  SmartPointer<std::vector<NumericType>> calculateVelocities(
      SmartPointer<viennals::PointData<NumericType>> rates,
      const std::vector<std::array<NumericType, 3>> &coordinates,
      const std::vector<NumericType> &materialIds) override {

    auto velocity =
        SmartPointer<std::vector<NumericType>>::New(materialIds.size(), 0.);

    for (std::size_t i = 0; i < velocity->size(); i++) {
      std::vector<NumericType> fluxes;
      for (auto &fluxDataLabel : fluxDataLabels_) {
        fluxes.push_back(rates->getScalarData(fluxDataLabel)->at(i));
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
              std::string dataLabel)
      : sourcePower_(sourcePower), meanEnergy_(meanEnergy),
        sigmaEnergy_(sigmaEnergy), thresholdEnergy_(thresholdEnergy),
        B_sp_(B_sp), thetaRMin_(thetaRMin), thetaRMax_(thetaRMax),
        inflectAngle_(inflectAngle), minAngle_(minAngle), n_(n),
        dataLabel_(dataLabel),
        A_(1. / (1. + n * (M_PI_2 / inflectAngle - 1.))) {}

  void surfaceCollision(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                        const Vec3D<NumericType> &geomNormal,
                        const unsigned int primID, const int,
                        viennaray::TracingData<NumericType> &localData,
                        const viennaray::TracingData<NumericType> *,
                        RNG &) override final {
    NumericType flux = rayWeight;

    if (B_sp_ >= 0.) {
      NumericType cosTheta = -DotProduct(rayDir, geomNormal);
      flux *= (1 + B_sp_ * (1 - cosTheta * cosTheta)) * cosTheta;
    }

    if (energy_ > 0.)
      flux *= std::max(std::sqrt(energy_) - std::sqrt(thresholdEnergy_),
                       NumericType(0.));

    localData.getVectorData(0)[primID] += flux;
  }
  std::pair<NumericType, Vec3D<NumericType>>
  surfaceReflection(NumericType, const Vec3D<NumericType> &rayDir,
                    const Vec3D<NumericType> &geomNormal, const unsigned int,
                    const int, const viennaray::TracingData<NumericType> *,
                    RNG &rngState) override final {

    auto cosTheta = -DotProduct(rayDir, geomNormal);
    assert(cosTheta >= 0 && "Hit backside of disc");
    assert(cosTheta <= 1 + 1e-6 && "Error in calculating cos theta");

    NumericType incomingAngle =
        std::acos(std::max(std::min(cosTheta, static_cast<NumericType>(1.)),
                           static_cast<NumericType>(0.)));

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
        rayDir, geomNormal, rngState, std::max(incomingAngle, minAngle_));

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
class NeutralParticle
    : public viennaray::Particle<NeutralParticle<NumericType, D>, NumericType> {
public:
  NeutralParticle(NumericType sticking, NumericType sourcePower,
                  std::string dataLabel)
      : stickingProbability_(sticking), sourcePower_(sourcePower),
        dataLabel_(dataLabel) {}

  void surfaceCollision(NumericType rayWeight, const Vec3D<NumericType> &,
                        const Vec3D<NumericType> &, const unsigned int primID,
                        const int,
                        viennaray::TracingData<NumericType> &localData,
                        const viennaray::TracingData<NumericType> *,
                        RNG &) override final {
    localData.getVectorData(0)[primID] += rayWeight;
  }
  std::pair<NumericType, Vec3D<NumericType>>
  surfaceReflection(NumericType, const Vec3D<NumericType> &rayDir,
                    const Vec3D<NumericType> &geomNormal, const unsigned int,
                    const int, const viennaray::TracingData<NumericType> *,
                    RNG &rngState) override final {
    auto direction =
        viennaray::ReflectionDiffuse<NumericType, D>(geomNormal, rngState);
    return std::pair<NumericType, Vec3D<NumericType>>{stickingProbability_,
                                                      direction};
  }
  void initNew(RNG &) override final {}
  NumericType getSourceDistributionPower() const override final {
    return sourcePower_;
  }
  std::vector<std::string> getLocalDataLabels() const override final {
    return {dataLabel_};
  }

private:
  const NumericType stickingProbability_;
  const NumericType sourcePower_;
  const std::string dataLabel_;
};
} // namespace impl

template <typename NumericType, int D>
class MultiParticleProcess : public ProcessModel<NumericType, D> {
public:
  MultiParticleProcess() {
    // surface model
    auto surfModel =
        SmartPointer<impl::MultiParticleSurfaceModel<NumericType, D>>::New(
            fluxDataLabels_);

    // velocity field
    auto velField = SmartPointer<DefaultVelocityField<NumericType>>::New(2);

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("MultiParticleProcess");
  }

  void addNeutralParticle(NumericType stickingProbability) {
    std::string dataLabel =
        "neutralFlux" + std::to_string(fluxDataLabels_.size());
    fluxDataLabels_.push_back(dataLabel);
    auto particle = std::make_unique<impl::NeutralParticle<NumericType, D>>(
        stickingProbability, 1., dataLabel);
    this->insertNextParticleType(particle);
  }

  void addIonParticle(NumericType sourcePower, NumericType thetaRMin = 0.,
                      NumericType thetaRMax = 90., NumericType minAngle = 0.,
                      NumericType B_sp = -1., NumericType meanEnergy = 0.,
                      NumericType sigmaEnergy = 0.,
                      NumericType thresholdEnergy = 0.,
                      NumericType inflectAngle = 0., NumericType n = 1) {
    std::string dataLabel = "ionFlux" + std::to_string(fluxDataLabels_.size());
    fluxDataLabels_.push_back(dataLabel);
    auto particle = std::make_unique<impl::IonParticle<NumericType, D>>(
        sourcePower, meanEnergy, sigmaEnergy, thresholdEnergy, B_sp,
        thetaRMin * M_PI / 180., thetaRMax * M_PI / 180.,
        inflectAngle * M_PI / 180., minAngle * M_PI / 180., n, dataLabel);
    this->insertNextParticleType(particle);
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
};

} // namespace viennaps
