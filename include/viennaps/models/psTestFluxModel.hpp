#pragma once

#include "../process/psProcessModel.hpp"
#include "../psMaterials.hpp"

#include <rayParticle.hpp>
#include <rayReflection.hpp>

namespace viennaps {

using namespace viennacore;

namespace impl {
template <typename NumericType, int D>
class TestFluxParticle
    : public viennaray::Particle<TestFluxParticle<NumericType, D>,
                                 NumericType> {
public:
  TestFluxParticle(NumericType sticking, NumericType sourcePower)
      : stickingProbability_(sticking), sourcePower_(sourcePower) {}

  void surfaceCollision(NumericType rayWeight, const Vec3D<NumericType> &,
                        const Vec3D<NumericType> &, const unsigned int primID,
                        const int,
                        viennaray::TracingData<NumericType> &localData,
                        const viennaray::TracingData<NumericType> *,
                        RNG &) override final {
    if (!isReflected_) {
      localData.getVectorData(0)[primID] += rayWeight;
    } else {
      localData.getVectorData(1)[primID] += rayWeight;
    }
  }
  std::pair<NumericType, Vec3D<NumericType>>
  surfaceReflection(NumericType, const Vec3D<NumericType> &,
                    const Vec3D<NumericType> &geomNormal, const unsigned int,
                    const int, const viennaray::TracingData<NumericType> *,
                    RNG &rngState) override final {
    isReflected_ = true;
    auto direction =
        viennaray::ReflectionDiffuse<NumericType, D>(geomNormal, rngState);
    return std::pair<NumericType, Vec3D<NumericType>>{stickingProbability_,
                                                      direction};
  }
  void initNew(RNG &) override final { isReflected_ = false; }
  NumericType getSourceDistributionPower() const override final {
    return sourcePower_;
  }
  std::vector<std::string> getLocalDataLabels() const override final {
    return {"primaryFlux", "secondaryFlux"};
  }

private:
  const NumericType stickingProbability_;
  const NumericType sourcePower_;
  bool isReflected_ = false;
};
} // namespace impl

#ifdef VIENNACORE_COMPILE_GPU
namespace gpu {

template <typename NumericType, int D>
class TestFluxModel final : public ProcessModelGPU<NumericType, D> {
public:
  TestFluxModel(NumericType stickingProbability, NumericType sourceExponent) {
    // particles
    viennaray::gpu::Particle<NumericType> particle{
        .name = "TestFluxParticle",
        .sticking = stickingProbability,
        .cosineExponent = sourceExponent};
    particle.dataLabels.push_back("primaryFlux");
    particle.dataLabels.push_back("secondaryFlux");

    std::unordered_map<std::string, unsigned> pMap = {{"TestFluxParticle", 0}};
    std::vector<viennaray::gpu::CallableConfig> cMap = {
        {0, viennaray::gpu::CallableSlot::COLLISION,
         "__direct_callable__testFluxCollision"},
        {0, viennaray::gpu::CallableSlot::REFLECTION,
         "__direct_callable__singleNeutralReflection"}};
    this->setParticleCallableMap(pMap, cMap);
    this->setCallableFileName("CallableWrapper");

    // surface model
    auto surfModel = SmartPointer<::viennaps::SurfaceModel<NumericType>>::New();

    // velocity field
    auto velField =
        SmartPointer<::viennaps::DefaultVelocityField<NumericType, D>>::New();

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->insertNextParticleType(particle);
    this->setProcessName("TestFluxModel");
    this->hasGPU = true;
  }
};
} // namespace gpu
#endif

// Etching or deposition based on a single particle model with diffuse
// reflections.
template <typename NumericType, int D>
class TestFluxModel : public ProcessModelCPU<NumericType, D> {
  NumericType stickingProbability_ = 1.;
  NumericType sourceDistributionPower_ = 1.;

public:
  TestFluxModel(NumericType stickingProbability = 1.,
                NumericType sourceDistributionPower = 1.)
      : stickingProbability_(stickingProbability),
        sourceDistributionPower_(sourceDistributionPower) {

    auto particle = std::make_unique<impl::TestFluxParticle<NumericType, D>>(
        stickingProbability_, sourceDistributionPower_);

    // surface model
    auto surfModel = SmartPointer<SurfaceModel<NumericType>>::New();

    // velocity field
    auto velField = SmartPointer<DefaultVelocityField<NumericType, D>>::New();

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->insertNextParticleType(particle);
    this->setProcessName("TestFluxModel");
    this->hasGPU = true;
  }

#ifdef VIENNACORE_COMPILE_GPU
  SmartPointer<ProcessModelBase<NumericType, D>> getGPUModel() final {
    return SmartPointer<gpu::TestFluxModel<NumericType, D>>::New(
        stickingProbability_, sourceDistributionPower_);
  }
#endif
};

} // namespace viennaps