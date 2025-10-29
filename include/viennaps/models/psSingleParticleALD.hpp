#pragma once

#include "../process/psProcessModel.hpp"

namespace viennaps {

using namespace viennacore;

namespace impl {

template <typename NumericType>
class SingleParticleALDSurfaceModel : public SurfaceModel<NumericType> {
  using SurfaceModel<NumericType>::coverages;

  const NumericType dt_;
  const NumericType gpc_;
  const NumericType s0_;

  const NumericType ev_;
  const NumericType flux_;
  const NumericType sticking_;

public:
  SingleParticleALDSurfaceModel(NumericType dt, NumericType gpc, NumericType ev,
                                NumericType flux, NumericType sticking,
                                NumericType s0)
      : dt_(dt), gpc_(gpc), s0_(s0), ev_(ev), flux_(flux), sticking_(sticking) {
  }

  void initializeCoverages(unsigned numPoints) override {
    if (coverages == nullptr) {
      coverages = viennals::PointData<NumericType>::New();
    } else {
      coverages->clear();
    }
    std::vector<NumericType> cov(numPoints, 0.);
    coverages->insertNextScalarData(cov, "Coverage");
  }

  SmartPointer<std::vector<NumericType>>
  calculateVelocities(SmartPointer<viennals::PointData<NumericType>> rates,
                      const std::vector<Vec3D<NumericType>> &coordinates,
                      const std::vector<NumericType> &materialIds) override {

    const auto numPoints = rates->getScalarData(0)->size();
    std::vector<NumericType> depoRate(numPoints, 0.);

    auto Coverage = coverages->getScalarData("Coverage");

    for (size_t i = 0; i < numPoints; ++i) {
      depoRate[i] = gpc_ * Coverage->at(i);
    }

    return SmartPointer<std::vector<NumericType>>::New(std::move(depoRate));
  }

  void updateCoverages(SmartPointer<viennals::PointData<NumericType>> rates,
                       const std::vector<NumericType> &materialIds) override {
    // update coverages based on fluxes
    const auto numPoints = materialIds.size();
    const auto ParticleFlux = rates->getScalarData("ParticleFlux");
    auto Coverage = coverages->getScalarData("Coverage");

    assert(rates->getScalarData("ParticleFlux")->size() == numPoints &&
           "ParticleFlux size mismatch");
    assert(coverages->getScalarData("Coverage")->size() == numPoints &&
           "Coverage size mismatch");

    for (size_t i = 0; i < numPoints; ++i) {
      Coverage->at(i) +=
          (flux_ * sticking_ * ParticleFlux->at(i) * (1 - Coverage->at(i)) -
           ev_ * Coverage->at(i)) *
          dt_ / s0_;

      Coverage->at(i) = std::min(Coverage->at(i), NumericType(1.0));
    }
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
public:
  SingleParticleALD(
      NumericType stickingProbability, // particle sticking probability
      int numCycles, // number of cycles to simulate in one advection step
      NumericType growthPerCycle, // growth per cycle
      int totalCycles,            // total number of cycles
      NumericType
          coverageTimeStep, // time step for solving the coverage equation
      NumericType evFlux,   // evaporation flux
      NumericType inFlux,   // incoming flux
      NumericType s0,       // saturation coverage
      NumericType gasMFP    // mean free path of the particles in the gas
  ) {
    if (gasMFP > 0) {
      Logger::getInstance()
          .addWarning(
              "Mean free path > 0 specified for GPU SingleParticleALD model. "
              "Currently only mean free path = 0 (ballistic transport) is "
              "supported.")
          .print();
    }
    viennaray::gpu::Particle<NumericType> particle{
        .name = "SingleParticleALD", .sticking = stickingProbability};
    particle.dataLabels.push_back("ParticleFlux");

    NumericType gpc = totalCycles / numCycles * growthPerCycle;

    // surface model
    auto surfModel =
        SmartPointer<::viennaps::impl::SingleParticleALDSurfaceModel<
            NumericType>>::New(coverageTimeStep, gpc, evFlux, inFlux,
                               stickingProbability, s0);

    // velocity field
    auto velField = SmartPointer<DefaultVelocityField<NumericType, D>>::New();

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->insertNextParticleType(particle);
    this->setProcessName("SingleParticleALD");
    this->isALP = true;
    this->hasGPU = true;

    // Callables
    /// TODO: Implement GPU callable functions for ALD particle-surface
    /// interactions

    this->processMetaData["stickingProbability"] =
        std::vector<double>{stickingProbability};
    this->processMetaData["numCycles"] =
        std::vector<double>{static_cast<double>(numCycles)};
    this->processMetaData["growthPerCycle"] =
        std::vector<double>{growthPerCycle};
    this->processMetaData["totalCycles"] =
        std::vector<double>{static_cast<double>(totalCycles)};
    this->processMetaData["coverageTimeStep"] =
        std::vector<double>{coverageTimeStep};
    this->processMetaData["evaporationFlux"] = std::vector<double>{evFlux};
    this->processMetaData["incomingFlux"] = std::vector<double>{inFlux};
    this->processMetaData["s0"] = std::vector<double>{s0};
    this->processMetaData["gasMeanFreePath"] = std::vector<double>{gasMFP};
  }
};
} // namespace gpu
#endif

template <typename NumericType, int D>
class SingleParticleALD : public ProcessModelCPU<NumericType, D> {
public:
  SingleParticleALD(
      NumericType stickingProbability, // particle sticking probability
      int numCycles, // number of cycles to simulate in one advection step
      NumericType growthPerCycle, // growth per cycle
      int totalCycles,            // total number of cycles
      NumericType
          coverageTimeStep, // time step for solving the coverage equation
      NumericType evFlux,   // evaporation flux
      NumericType inFlux,   // incoming flux
      NumericType s0,       // saturation coverage
      NumericType gasMFP    // mean free path of the particles in the gas
  ) {
    auto particle =
        std::make_unique<impl::SingleParticleALDParticle<NumericType, D>>(
            stickingProbability, gasMFP);

    NumericType gpc = totalCycles / numCycles * growthPerCycle;

    // surface model
    auto surfModel =
        SmartPointer<impl::SingleParticleALDSurfaceModel<NumericType>>::New(
            coverageTimeStep, gpc, evFlux, inFlux, stickingProbability, s0);

    // velocity field
    auto velField = SmartPointer<DefaultVelocityField<NumericType, D>>::New();

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->insertNextParticleType(particle);
    this->setProcessName("SingleParticleALD");
    this->isALP = true;
    this->hasGPU = true;

    this->processMetaData["stickingProbability"] =
        std::vector<double>{stickingProbability};
    this->processMetaData["numCycles"] =
        std::vector<double>{static_cast<double>(numCycles)};
    this->processMetaData["growthPerCycle"] =
        std::vector<double>{growthPerCycle};
    this->processMetaData["totalCycles"] =
        std::vector<double>{static_cast<double>(totalCycles)};
    this->processMetaData["coverageTimeStep"] =
        std::vector<double>{coverageTimeStep};
    this->processMetaData["evaporationFlux"] = std::vector<double>{evFlux};
    this->processMetaData["incomingFlux"] = std::vector<double>{inFlux};
    this->processMetaData["s0"] = std::vector<double>{s0};
    this->processMetaData["gasMeanFreePath"] = std::vector<double>{gasMFP};
  }

#ifdef VIENNACORE_COMPILE_GPU
  SmartPointer<ProcessModelBase<NumericType, D>> getGPUModel() override {
    auto gpuModel = SmartPointer<gpu::SingleParticleALD<NumericType, D>>::New(
        this->processMetaData["stickingProbability"][0],
        static_cast<int>(this->processMetaData["numCycles"][0]),
        this->processMetaData["growthPerCycle"][0],
        static_cast<int>(this->processMetaData["totalCycles"][0]),
        this->processMetaData["coverageTimeStep"][0],
        this->processMetaData["evaporationFlux"][0],
        this->processMetaData["incomingFlux"][0],
        this->processMetaData["s0"][0],
        this->processMetaData["gasMeanFreePath"][0]);
    return gpuModel;
  }
#endif
};

PS_PRECOMPILE_PRECISION_DIMENSION(SingleParticleALD)

} // namespace viennaps
