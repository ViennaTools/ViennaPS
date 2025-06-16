#pragma once

#include <psProcessModel.hpp>

namespace viennaps {

using namespace viennacore;

namespace impl {

template <typename NumericType, int D>
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

      Coverage->at(i) = std::min(Coverage->at(i), 1.);
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
    NumericType S_eff = beta * (1. - phi);

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

template <typename NumericType, int D>
class SingleParticleALD : public ProcessModel<NumericType, D> {
public:
  SingleParticleALD(
      const NumericType stickingProbability, // particle sticking probability
      const int numCycles, // number of cycles to simulate in one advection step
      const NumericType growthPerCycle, // growth per cycle
      const int totalCycles,            // total number of cycles
      const NumericType
          coverageTimeStep,     // time step for solving the coverage equation
      const NumericType evFlux, // evaporation flux
      const NumericType inFlux, // incoming flux
      const NumericType s0,     // saturation coverage
      const NumericType gasMFP  // mean free path of the particles in the gas
  ) {
    auto particle =
        std::make_unique<impl::SingleParticleALDParticle<NumericType, D>>(
            stickingProbability, gasMFP);

    NumericType gpc = totalCycles / numCycles * growthPerCycle;

    // surface model
    auto surfModel =
        SmartPointer<impl::SingleParticleALDSurfaceModel<NumericType, D>>::New(
            coverageTimeStep, gpc, evFlux, inFlux, stickingProbability, s0);

    // velocity field
    auto velField = SmartPointer<DefaultVelocityField<NumericType, D>>::New(2);

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->insertNextParticleType(particle);
    this->setProcessName("SingleParticleALD");
  }
};

} // namespace viennaps
