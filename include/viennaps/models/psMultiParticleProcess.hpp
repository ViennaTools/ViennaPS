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
class MultiParticle
    : public viennaray::Particle<MultiParticle<NumericType, D>, NumericType> {
public:
  MultiParticle(NumericType sticking, NumericType sourcePower,
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

// Etching or deposition based on a single particle model with diffuse
// reflections.
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

  void addNeutralParticle(NumericType sticking, NumericType sourcePower,
                          std::string dataLabel) {
    auto particle = std::make_unique<impl::MultiParticle<NumericType, D>>(
        sticking, sourcePower, dataLabel);
    this->insertNextParticleType(particle);
    fluxDataLabels_.push_back(dataLabel);
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
