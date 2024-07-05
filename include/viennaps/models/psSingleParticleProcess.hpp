#pragma once

#include "../psMaterials.hpp"
#include "../psProcessModel.hpp"

#include <rayParticle.hpp>
#include <rayReflection.hpp>

namespace viennaps {

using namespace viennacore;

namespace impl {
template <typename NumericType, int D>
class SingleParticleSurfaceModel : public viennaps::SurfaceModel<NumericType> {
  const NumericType rateFactor_;
  const std::vector<Material> maskMaterials_;

public:
  SingleParticleSurfaceModel(NumericType rate,
                             const std::vector<Material> &mask)
      : rateFactor_(rate), maskMaterials_(mask) {}

  SmartPointer<std::vector<NumericType>> calculateVelocities(
      SmartPointer<viennals::PointData<NumericType>> rates,
      const std::vector<std::array<NumericType, 3>> &coordinates,
      const std::vector<NumericType> &materialIds) override {

    auto velocity =
        SmartPointer<std::vector<NumericType>>::New(materialIds.size(), 0.);
    auto flux = rates->getScalarData("particleFlux");

    for (std::size_t i = 0; i < velocity->size(); i++) {
      if (!isMaskMaterial(materialIds[i])) {
        velocity->at(i) = flux->at(i) * rateFactor_;
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
class SingleParticle
    : public viennaray::Particle<SingleParticle<NumericType, D>, NumericType> {
public:
  SingleParticle(NumericType sticking, NumericType sourcePower)
      : stickingProbability_(sticking), sourcePower_(sourcePower) {}

  void surfaceCollision(NumericType rayWeight, const Vec3D<NumericType> &,
                        const Vec3D<NumericType> &, const unsigned int primID,
                        const int,
                        viennaray::TracingData<NumericType> &localData,
                        const viennaray::TracingData<NumericType> *,
                        RNG &) override final {
    localData.getVectorData(0)[primID] += rayWeight;
  }
  std::pair<NumericType, Vec3D<NumericType>>
  surfaceReflection(NumericType, const Vec3D<NumericType> &,
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
    return {"particleFlux"};
  }

private:
  const NumericType stickingProbability_;
  const NumericType sourcePower_;
};
} // namespace impl

// Etching or deposition based on a single particle model with diffuse
// reflections.
template <typename NumericType, int D> class SingleParticleProcess {
  SmartPointer<ProcessModel<NumericType, D>> processModel;

public:
  SingleParticleProcess(NumericType rate = 1.,
                        NumericType stickingProbability = 1.,
                        NumericType sourceDistributionPower = 1.,
                        Material maskMaterial = Material::None) {
    processModel = SmartPointer<ProcessModel<NumericType, D>>::New();
    std::vector<Material> maskMaterialVec = {maskMaterial};
    initialize(rate, stickingProbability, sourceDistributionPower,
               std::move(maskMaterialVec));
  }

  SingleParticleProcess(NumericType rate, NumericType stickingProbability,
                        NumericType sourceDistributionPower,
                        std::vector<Material> maskMaterial) {
    processModel = SmartPointer<ProcessModel<NumericType, D>>::New();
    initialize(rate, stickingProbability, sourceDistributionPower,
               std::move(maskMaterial));
  }

  auto getProcessModel() { return processModel; }

private:
  void initialize(NumericType rate, NumericType stickingProbability,
                  NumericType sourceDistributionPower,
                  std::vector<Material> &&maskMaterial) {
    // particles
    auto particle = std::make_unique<impl::SingleParticle<NumericType, D>>(
        stickingProbability, sourceDistributionPower);

    // surface model
    auto surfModel =
        SmartPointer<impl::SingleParticleSurfaceModel<NumericType, D>>::New(
            rate, maskMaterial);

    // velocity field
    auto velField = SmartPointer<DefaultVelocityField<NumericType>>::New(2);

    processModel->setSurfaceModel(surfModel);
    processModel->setVelocityField(velField);
    processModel->insertNextParticleType(particle);
    processModel->setProcessName("SingleParticleProcess");
  }
};

} // namespace viennaps
