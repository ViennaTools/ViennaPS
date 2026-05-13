#pragma once

#include "../process/psProcessModel.hpp"

namespace viennaps {

using namespace viennacore;

struct SingleParticleALDParams {
  double stickingProbability;  // particle sticking probability
  double gasMeanFreePath = -1; // mean free path of the particles in the gas

  double growthPerCycle; // growth per cycle
  int totalCycles;       // number of *real* cycles
  int numCycles;         // number of cycles to simulate

  double evaporationFlux; // evaporation flux
  double incomingFlux;    // incoming flux
  double s0;              // surface site density
  double coverageDiffusionCoefficient =
      0;                     // surface diffusion coefficient for coverage
  double purgePulseTime = 0; // time of purge pulse after each cycle

  auto toProcessMetaData() const {
    std::unordered_map<std::string, std::vector<double>> processData;

    processData["stickingProbability"] = {stickingProbability};
    processData["gasMeanFreePath"] = {gasMeanFreePath};
    processData["growthPerCycle"] = {growthPerCycle};
    processData["totalCycles"] = {static_cast<double>(totalCycles)};
    processData["numCycles"] = {static_cast<double>(numCycles)};
    processData["evaporationFlux"] = {evaporationFlux};
    processData["incomingFlux"] = {incomingFlux};
    processData["s0"] = {s0};
    processData["coverageDiffusionCoefficient"] = {
        coverageDiffusionCoefficient};
    processData["purgePulseTime"] = {purgePulseTime};

    return processData;
  }
};

namespace impl {

template <typename NumericType>
class SingleParticleALDSurfaceModel : public SurfaceModel<NumericType> {
  using SurfaceModel<NumericType>::coverages;

  NumericType dt_ = 0.0;
  SingleParticleALDParams const &params_;

public:
  SingleParticleALDSurfaceModel(SingleParticleALDParams const &params)
      : params_(params) {}

  void initializeCoverages(unsigned numPoints) override {
    if (coverages == nullptr) {
      coverages = viennals::PointData<NumericType>::New();
    } else {
      coverages->clear();
    }
    std::vector<NumericType> cov(numPoints, 0.);
    coverages->insertNextScalarData(cov, "Coverage");
  }

  void setTimeStep(NumericType dt) override { dt_ = dt; }

  SmartPointer<std::vector<NumericType>>
  calculateVelocities(SmartPointer<viennals::PointData<NumericType>> fluxes,
                      const std::vector<Vec3D<NumericType>> &coordinates,
                      const std::vector<NumericType> &materialIds) override {

    const auto numPoints = coordinates.size();
    std::vector<NumericType> depoRate(numPoints, 0.);

    auto Coverage = coverages->getScalarData("Coverage");
    assert(Coverage && Coverage->size() == numPoints);

    const NumericType gpc =
        params_.totalCycles / params_.numCycles * params_.growthPerCycle;

    for (std::size_t i = 0; i < numPoints; ++i) {
      depoRate[i] = gpc * Coverage->at(i);
    }

    return SmartPointer<std::vector<NumericType>>::New(std::move(depoRate));
  }

  void updateCoverages(SmartPointer<viennals::PointData<NumericType>> fluxes,
                       const std::vector<NumericType> &materialIds) override {
    // update coverages based on fluxes
    const auto numPoints = materialIds.size();
    const auto ParticleFlux = fluxes->getScalarData("ParticleFlux");
    auto Coverage = coverages->getScalarData("Coverage");

    assert(ParticleFlux && ParticleFlux->size() == numPoints &&
           "ParticleFlux size mismatch");
    assert(Coverage && Coverage->size() == numPoints &&
           "Coverage size mismatch");

    for (size_t i = 0; i < numPoints; ++i) {
      Coverage->at(i) += (params_.incomingFlux * params_.stickingProbability *
                              ParticleFlux->at(i) * (1 - Coverage->at(i)) -
                          params_.evaporationFlux * Coverage->at(i)) *
                         dt_ / params_.s0;

      Coverage->at(i) =
          std::clamp(Coverage->at(i), NumericType(0.0), NumericType(1.0));
    }
  }

  void updateCoveragesFromDesorption(
      SmartPointer<viennals::PointData<NumericType>> desorptionFluxes,
      const std::vector<NumericType> &materialIds) override {
    const auto numPoints = materialIds.size();
    const auto ParticleFlux = desorptionFluxes->getScalarData("ParticleFlux");
    auto Coverage = coverages->getScalarData("Coverage");

    assert(ParticleFlux && ParticleFlux->size() == numPoints &&
           "ParticleFlux size mismatch");
    assert(Coverage && Coverage->size() == numPoints &&
           "Coverage size mismatch");

    for (size_t i = 0; i < numPoints; ++i) {
      // desorption reduces coverage, while re-adsorption of desorbed particles
      // increases coverage
      Coverage->at(i) -= params_.evaporationFlux * Coverage->at(i) *
                         params_.purgePulseTime / params_.s0;

      Coverage->at(i) +=
          (params_.evaporationFlux * params_.stickingProbability *
           ParticleFlux->at(i) * (1 - Coverage->at(i))) *
          params_.purgePulseTime / params_.s0;

      Coverage->at(i) =
          std::clamp(Coverage->at(i), NumericType(0.0), NumericType(1.0));
    }
  }

  std::optional<std::unordered_map<std::string, NumericType>>
  getDiffusionCoefficients() const override {
    if (params_.coverageDiffusionCoefficient <= 0)
      return std::nullopt;
    return std::make_optional<std::unordered_map<std::string, NumericType>>(
        {{"Coverage", params_.coverageDiffusionCoefficient}});
  }

  std::optional<std::vector<NumericType>> getDesorptionWeights(
      const std::vector<NumericType> &materialIds) const override {
    std::vector<NumericType> desorptionWeights(materialIds.size(), 0.);
    if (params_.evaporationFlux <= 0)
      return desorptionWeights;

    if (coverages == nullptr)
      return desorptionWeights;

    auto Coverage = coverages->getScalarData("Coverage");
    if (!Coverage)
      return desorptionWeights;

    assert(Coverage->size() == materialIds.size() && "Coverage size mismatch");
    for (size_t i = 0; i < materialIds.size(); ++i) {
      desorptionWeights[i] = params_.evaporationFlux * Coverage->at(i) *
                             params_.purgePulseTime / params_.s0;
    }

    return desorptionWeights;
  }
};

template <typename NumericType, int D>
class SingleParticleALDParticle
    : public viennaray::Particle<SingleParticleALDParticle<NumericType, D>,
                                 NumericType> {
  const NumericType beta;
  const NumericType lambda;

public:
  SingleParticleALDParticle(const NumericType sticking,
                            const NumericType gasMFP)
      : beta(sticking), lambda(gasMFP) {}

  void surfaceCollision(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                        const Vec3D<NumericType> &geomNormal,
                        const unsigned int primID, const int materialId,
                        viennaray::TracingData<NumericType> &localData,
                        const viennaray::TracingData<NumericType> *globalData,
                        RNG &Rng) override final {
    localData.getVectorData(0)[primID] += rayWeight;
  }
  std::pair<NumericType, Vec3D<NumericType>>
  surfaceReflection(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                    const Vec3D<NumericType> &geomNormal,
                    const unsigned int primID, const int materialId,
                    const viennaray::TracingData<NumericType> *globalData,
                    RNG &Rng) override final {
    assert(primID < globalData->getVectorData(1).size() &&
           "PrimID out of bounds");

    // H2O surface coverage
    const auto &phi = globalData->getVectorData(0)[primID];
    // Obtain the sticking probability
    NumericType S_eff = beta * std::max(NumericType(1.) - phi, NumericType(0.));

    auto direction =
        viennaray::ReflectionDiffuse<NumericType, D>(geomNormal, Rng);
    return std::pair<NumericType, Vec3D<NumericType>>{S_eff, direction};
  }
  NumericType getSourceDistributionPower() const override final { return 1.; }
  std::vector<std::string> getLocalDataLabels() const override final {
    return {"ParticleFlux"};
  }
  NumericType getMeanFreePath() const override final { return lambda; }
};
} // namespace impl

#ifdef VIENNACORE_COMPILE_GPU
namespace gpu {
template <typename NumericType, int D>
class SingleParticleALD : public ProcessModelGPU<NumericType, D> {
  SingleParticleALDParams params_;

public:
  SingleParticleALD(const SingleParticleALDParams &params) : params_(params) {
    if (params_.gasMeanFreePath > 0) {
      VIENNACORE_LOG_WARNING(
          "Mean free path > 0 specified for GPU SingleParticleALD model. "
          "Currently only ballistic transport is supported.");
      params_.gasMeanFreePath = -1;
    }

    // particles
    viennaray::gpu::Particle<NumericType> particle{
        .name = "SingleParticle", .sticking = params_.stickingProbability};
    particle.dataLabels.push_back("ParticleFlux");

    std::unordered_map<std::string, unsigned> pMap = {{"SingleParticle", 0}};
    std::vector<viennaray::gpu::CallableConfig> cMap = {
        {0, viennaray::gpu::CallableSlot::COLLISION,
         "__direct_callable__singleNeutralCollision"},
        {0, viennaray::gpu::CallableSlot::REFLECTION,
         "__direct_callable__singleALDNeutralReflection"}};
    this->setParticleCallableMap(pMap, cMap);

    // surface model
    auto surfModel =
        SmartPointer<::viennaps::impl::SingleParticleALDSurfaceModel<
            NumericType>>::New(params_);

    // velocity field
    auto velField = SmartPointer<DefaultVelocityField<NumericType, D>>::New();

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->insertNextParticleType(particle);
    this->setProcessName("SingleParticleALD");
    this->isALP = true;
    this->hasGPU = true;

    this->processMetaData = params_.toProcessMetaData();
  }
};
} // namespace gpu
#endif

template <typename NumericType, int D>
class SingleParticleALD : public ProcessModelCPU<NumericType, D> {
  SingleParticleALDParams params_;

public:
  SingleParticleALD(const SingleParticleALDParams &params) : params_(params) {
    auto particle =
        std::make_unique<impl::SingleParticleALDParticle<NumericType, D>>(
            params_.stickingProbability, params_.gasMeanFreePath);

    // surface model
    auto surfModel =
        SmartPointer<impl::SingleParticleALDSurfaceModel<NumericType>>::New(
            params_);

    // velocity field
    auto velField = SmartPointer<DefaultVelocityField<NumericType, D>>::New();

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->insertNextParticleType(particle);
    this->setProcessName("SingleParticleALD");
    this->isALP = true;
    this->hasGPU = true;

    this->processMetaData = params_.toProcessMetaData();
  }

#ifdef VIENNACORE_COMPILE_GPU
  SmartPointer<ProcessModelBase<NumericType, D>> getGPUModel() override {
    auto model =
        SmartPointer<gpu::SingleParticleALD<NumericType, D>>::New(params_);
    model->setProcessName(this->getProcessName().value());
    return model;
  }
#endif
};

PS_PRECOMPILE_PRECISION_DIMENSION(SingleParticleALD)

} // namespace viennaps
