#pragma once

#include "../materials/psMaterialMap.hpp"
#include "../process/psProcessModel.hpp"
#include "../psConstants.hpp"
#include "../psUnits.hpp"

#include <rayParticle.hpp>
#include <rayPointNeighborhood.hpp>
#include <rayReflection.hpp>
#include <vcKDTree.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace viennaps {

using namespace viennacore;

template <typename NumericType> struct NeutralTransportParameters {
  // Gas-phase incoming neutral flux scale in 10^20 molecules / (m^2 s). The ray
  // tracer provides the local dimensionless transport factor.
  NumericType incomingFlux = 1.;

  // Fixed-pressure reservoir boundary condition used by the benchmark paper.
  NumericType sourcePressure = 1.;          // Pa
  NumericType sourceTemperature = 300.;     // K
  NumericType sourceMolecularMass = 18.998; // amu, atomic fluorine

  // Langmuir adsorption/desorption parameters.
  NumericType zeroCoverageSticking = 0.1;
  NumericType etchFrontSticking = 1.;
  NumericType desorptionRate = 0.; // 1 / s
  Material desorptionMaterial = Material::Mask;
  NumericType kEtch = 0.;                   // 1 / s
  NumericType surfaceSiteDensity = 1.66e-5; // mol / m^2
  NumericType siliconDensity = 8.3e4;       // mol / m^3
  NumericType coverageTimeStep = 1.;        // s
  bool useSteadyStateCoverage = true;

  // Surface diffusion for Eq. (9) in the neutral transport paper. The
  // coefficient is expressed in the active length unit squared per second.
  NumericType surfaceDiffusionCoefficient = 0.;
  Material surfaceDiffusionMaterial = Material::Mask;
  NumericType surfaceDiffusionNeighborDistance = 0.;
  NumericType surfaceDiffusionSolverTolerance = 1e-8;
  unsigned surfaceDiffusionMaxIterations = 500;

  // Feature-scale process parameters.
  NumericType sourceDistributionPower = 1.;
  Material etchFrontMaterial = Material::Si;

  std::string fluxLabel = "neutralFlux";
  std::string coverageLabel = "neutralCoverage";
};

#ifdef VIENNACORE_COMPILE_GPU
struct NeutralTransportParametersGPU {
  NeutralTransportParametersGPU() = default;
  template <typename NumericType>
  explicit NeutralTransportParametersGPU(
      const NeutralTransportParameters<NumericType> &p)
      : etchFrontSticking(static_cast<float>(p.etchFrontSticking)),
        zeroCoverageSticking(static_cast<float>(p.zeroCoverageSticking)),
        etchFrontMaterialId(static_cast<int>(p.etchFrontMaterial.legacyId())),
        desorptionRate(static_cast<float>(p.desorptionRate)),
        kEtch(static_cast<float>(p.kEtch)),
        surfaceSiteDensity(static_cast<float>(p.surfaceSiteDensity)),
        siliconDensity(static_cast<float>(p.siliconDensity)),
        incomingFlux(static_cast<float>(p.incomingFlux)) {}

  float etchFrontSticking = 1.f;
  float zeroCoverageSticking = 0.1f;
  int etchFrontMaterialId = 10; // Material::Si.legacyId()
  float desorptionRate = 0.f;
  float kEtch = 0.f;
  float surfaceSiteDensity = 1.66e-5f;
  float siliconDensity = 8.3e4f;
  float incomingFlux = 1.f;
};
#endif

namespace impl {

template <typename NumericType, int D>
class NeutralTransportSurfaceModel : public SurfaceModel<NumericType> {
public:
  using SurfaceModel<NumericType>::coverages;
  using SurfaceModel<NumericType>::surfaceData;

  explicit NeutralTransportSurfaceModel(
      const NeutralTransportParameters<NumericType> &pParams)
      : params(pParams) {}

  void initializeCoverages(unsigned numGeometryPoints) override {
    if (coverages == nullptr) {
      coverages = viennals::PointData<NumericType>::New();
    } else {
      coverages->clear();
    }

    std::vector<NumericType> coverage(numGeometryPoints, 0.);
    coverages->insertNextScalarData(std::move(coverage), params.coverageLabel);
  }

  void initializeSurfaceData(unsigned numGeometryPoints) override {
    if (!Logger::hasIntermediate())
      return;

    if (surfaceData == nullptr) {
      surfaceData = viennals::PointData<NumericType>::New();
    } else {
      surfaceData->clear();
    }

    std::vector<NumericType> sticking(numGeometryPoints, 0.);
    surfaceData->insertNextScalarData(std::move(sticking),
                                      "stickingCoefficient");
  }

  void setSurfaceCoordinates(
      const std::vector<Vec3D<NumericType>> &coordinates) override {
    if (cachedCoordinates_.size() != coordinates.size() ||
        !std::equal(cachedCoordinates_.begin(), cachedCoordinates_.end(),
                    coordinates.begin())) {
      diffusionGraphValid_ = false;
    }
    cachedCoordinates_ = coordinates;
    hasCoordinates_ = true;
  }

  SmartPointer<std::vector<NumericType>>
  calculateVelocities(SmartPointer<viennals::PointData<NumericType>> fluxes,
                      const std::vector<Vec3D<NumericType>> &coordinates,
                      const std::vector<NumericType> &materialIds) override {
    static_cast<void>(coordinates);
    auto neutralFlux = fluxes->getScalarData(params.fluxLabel);
    assert(neutralFlux && neutralFlux->size() == materialIds.size());

    auto coverage = coverages->getScalarData(params.coverageLabel);
    assert(coverage && coverage->size() == materialIds.size());

    auto velocity =
        SmartPointer<std::vector<NumericType>>::New(materialIds.size(), 0.);

#pragma omp parallel for
    for (size_t i = 0; i < materialIds.size(); ++i) {
      if (MaterialMap::isMaterial(materialIds[i], params.etchFrontMaterial)) {
        const auto etchVelocity =
            params.siliconDensity > 0.
                ? params.kEtch * params.surfaceSiteDensity * coverage->at(i) /
                      params.siliconDensity
                : NumericType(0.);
        velocity->at(i) = -etchVelocity * units::Time::convertSecond() /
                          units::Length::convertMeter();
      }
    }

    if (Logger::hasIntermediate() && surfaceData != nullptr) {
      auto stickingData = surfaceData->getScalarData("stickingCoefficient");
      if (stickingData && stickingData->size() == coverage->size()) {
#pragma omp parallel for
        for (size_t i = 0; i < coverage->size(); ++i) {
          if (MaterialMap::isMaterial(materialIds[i],
                                      params.etchFrontMaterial)) {
            stickingData->at(i) = params.etchFrontSticking;
          } else {
            stickingData->at(i) =
                params.zeroCoverageSticking * (1. - coverage->at(i));
          }
        }
      }
    }

    return velocity;
  }

  void updateCoverages(SmartPointer<viennals::PointData<NumericType>> fluxes,
                       const std::vector<NumericType> &materialIds) override {
    auto neutralFlux = fluxes->getScalarData(params.fluxLabel);
    auto coverage = coverages->getScalarData(params.coverageLabel);
    assert(neutralFlux && coverage);
    assert(neutralFlux->size() == materialIds.size());

    coverage->resize(neutralFlux->size());

    std::vector<NumericType> *stickingData = nullptr;
    if (Logger::hasIntermediate() && surfaceData != nullptr) {
      stickingData = surfaceData->getScalarData("stickingCoefficient");
      if (stickingData)
        stickingData->resize(neutralFlux->size());
    }

#pragma omp parallel for
    for (size_t i = 0; i < neutralFlux->size(); ++i) {
      const auto sticking =
          MaterialMap::isMaterial(materialIds[i], params.etchFrontMaterial)
              ? params.etchFrontSticking
              : params.zeroCoverageSticking;
      const auto adsorptionCoefficient =
          params.surfaceSiteDensity > 0.
              ? sticking * params.incomingFlux * neutralFlux->at(i) /
                    (constants::N_A * params.surfaceSiteDensity)
              : NumericType(0.);
      const auto etchLossRate =
          MaterialMap::isMaterial(materialIds[i], params.etchFrontMaterial)
              ? params.kEtch
              : NumericType(0.);
      const auto desorptionLossRate =
          MaterialMap::isMaterial(materialIds[i], params.desorptionMaterial)
              ? params.desorptionRate
              : NumericType(0.);
      const auto coverageLossRate = desorptionLossRate + etchLossRate;

      if (params.useSteadyStateCoverage) {
        const auto denominator = adsorptionCoefficient + coverageLossRate;
        coverage->at(i) =
            denominator > 0. ? adsorptionCoefficient / denominator : 0.;
      } else {
        coverage->at(i) += params.coverageTimeStep *
                           (adsorptionCoefficient * (1. - coverage->at(i)) -
                            coverageLossRate * coverage->at(i));
      }

      coverage->at(i) =
          std::clamp(coverage->at(i), NumericType(0.), NumericType(1.));
    }

    if (params.surfaceDiffusionCoefficient > 0. && hasCoordinates_) {
      applySurfaceDiffusion(cachedCoordinates_, materialIds, *coverage);
    }

    if (stickingData) {
#pragma omp parallel for
      for (size_t i = 0; i < coverage->size(); ++i) {
        if (MaterialMap::isMaterial(materialIds[i], params.etchFrontMaterial)) {
          stickingData->at(i) = params.etchFrontSticking;
        } else {
          stickingData->at(i) =
              params.zeroCoverageSticking * (1. - coverage->at(i));
        }
      }
    }
  }

  std::vector<NumericType> getDesorptionWeights(
      const std::vector<NumericType> &materialIds) const override {
    if (params.desorptionRate <= 0. || params.surfaceSiteDensity <= 0. ||
        params.incomingFlux <= 0. || coverages == nullptr) {
      return {};
    }

    const auto coverage = coverages->getScalarData(params.coverageLabel);
    if (coverage == nullptr || coverage->size() != materialIds.size()) {
      return {};
    }

    std::vector<NumericType> weights(materialIds.size(), 0.);
    bool hasNonZeroWeight = false;

#pragma omp parallel for
    for (size_t i = 0; i < materialIds.size(); ++i) {
      if (!MaterialMap::isMaterial(materialIds[i], params.desorptionMaterial)) {
        continue;
      }
      const auto theta =
          std::clamp(coverage->at(i), NumericType(0.), NumericType(1.));
      weights[i] = params.desorptionRate * theta * params.surfaceSiteDensity *
                   constants::N_A / params.incomingFlux;
    }

    for (const auto weight : weights) {
      if (weight > NumericType(0.)) {
        hasNonZeroWeight = true;
        break;
      }
    }

    return hasNonZeroWeight ? weights : std::vector<NumericType>{};
  }

private:
  struct DiffusionEdge {
    size_t index;
    NumericType weight;
  };

  static NumericType squaredDistance(const Vec3D<NumericType> &a,
                                     const Vec3D<NumericType> &b) {
    NumericType result = 0.;
    for (int i = 0; i < D; ++i) {
      const auto diff = a[i] - b[i];
      result += diff * diff;
    }
    return result;
  }

  static NumericType dotActive(const std::vector<NumericType> &a,
                               const std::vector<NumericType> &b,
                               const std::vector<char> &active) {
    NumericType result = 0.;
#pragma omp parallel for reduction(+ : result)
    for (size_t i = 0; i < active.size(); ++i) {
      if (active[i]) {
        result += a[i] * b[i];
      }
    }
    return result;
  }

  NumericType estimateNeighborDistance(
      const std::vector<Vec3D<NumericType>> &coordinates) const {
    if (params.surfaceDiffusionNeighborDistance > NumericType(0.)) {
      return params.surfaceDiffusionNeighborDistance;
    }

    NumericType minPositiveDistance2 = std::numeric_limits<NumericType>::max();
    NumericType sumNearestDistance = 0.;
    size_t nearestCount = 0;

    KDTree<NumericType, Vec3D<NumericType>> tree;
    tree.setPoints(coordinates);
    tree.build();

    const int numNearest = std::min<int>(4, coordinates.size());
    for (size_t i = 0; i < coordinates.size(); ++i) {
      const auto nearest = tree.findKNearest(coordinates[i], numNearest);
      if (!nearest) {
        continue;
      }

      NumericType nearestDistance = std::numeric_limits<NumericType>::max();
      for (const auto &[index, distance] : *nearest) {
        if (index != i &&
            distance > std::numeric_limits<NumericType>::epsilon()) {
          nearestDistance = std::min(nearestDistance, distance);
          minPositiveDistance2 =
              std::min(minPositiveDistance2, distance * distance);
        }
      }
      if (nearestDistance < std::numeric_limits<NumericType>::max()) {
        sumNearestDistance += nearestDistance;
        ++nearestCount;
      }
    }

    if (nearestCount == 0) {
      return NumericType(0.);
    }

    const auto averageNearestDistance =
        sumNearestDistance / static_cast<NumericType>(nearestCount);
    return std::max(NumericType(1.75) * averageNearestDistance,
                    NumericType(1.01) * std::sqrt(minPositiveDistance2));
  }

  void buildDiffusionGraph(
      const std::vector<Vec3D<NumericType>> &coordinates) const {
    diffusionGraph_.clear();
    diffusionGraph_.resize(coordinates.size());
    diffusionNeighborDistance_ = estimateNeighborDistance(coordinates);
    if (diffusionNeighborDistance_ <= NumericType(0.)) {
      diffusionGraphValid_ = true;
      return;
    }

    Vec3D<NumericType> minExtent = coordinates.front();
    Vec3D<NumericType> maxExtent = coordinates.front();
    for (const auto &point : coordinates) {
      for (int i = 0; i < D; ++i) {
        minExtent[i] = std::min(minExtent[i], point[i]);
        maxExtent[i] = std::max(maxExtent[i], point[i]);
      }
    }

    viennaray::PointNeighborhood<NumericType, D> neighborhood;
    neighborhood.template init<3>(coordinates, diffusionNeighborDistance_,
                                  minExtent, maxExtent);

    const auto epsilon = std::numeric_limits<NumericType>::epsilon();
    for (size_t i = 0; i < coordinates.size(); ++i) {
      for (const auto neighborIndex : neighborhood.getNeighborIndices(i)) {
        const size_t j = neighborIndex;
        if (j <= i) {
          continue;
        }
        const auto distance2 = squaredDistance(coordinates[i], coordinates[j]);
        if (distance2 <= epsilon) {
          continue;
        }
        const auto weight = NumericType(1.) / distance2;
        diffusionGraph_[i].push_back({j, weight});
        diffusionGraph_[j].push_back({i, weight});
      }
    }

    diffusionGraphValid_ = true;
  }

  void applyDiffusionMatrix(const std::vector<NumericType> &x,
                            std::vector<NumericType> &result,
                            const std::vector<char> &active,
                            NumericType diffusionStep) const {
    result.assign(x.size(), NumericType(0.));
#pragma omp parallel for
    for (size_t i = 0; i < x.size(); ++i) {
      if (!active[i]) {
        result[i] = x[i];
        continue;
      }

      NumericType value = x[i];
      for (const auto &edge : diffusionGraph_[i]) {
        if (active[edge.index]) {
          value += diffusionStep * edge.weight * (x[i] - x[edge.index]);
        }
      }
      result[i] = value;
    }
  }

  void applySurfaceDiffusion(const std::vector<Vec3D<NumericType>> &coordinates,
                             const std::vector<NumericType> &materialIds,
                             std::vector<NumericType> &coverage) const {
    if (coordinates.size() != coverage.size() ||
        materialIds.size() != coverage.size() ||
        params.coverageTimeStep <= NumericType(0.)) {
      return;
    }

    if (!diffusionGraphValid_ || diffusionGraph_.size() != coordinates.size()) {
      buildDiffusionGraph(coordinates);
    }
    if (diffusionGraph_.empty()) {
      return;
    }

    std::vector<char> active(coverage.size(), false);
    size_t activeCount = 0;
    for (size_t i = 0; i < materialIds.size(); ++i) {
      active[i] = MaterialMap::isMaterial(materialIds[i],
                                          params.surfaceDiffusionMaterial);
      if (active[i]) {
        ++activeCount;
      }
    }
    if (activeCount < 2) {
      return;
    }

    const auto diffusionStep =
        params.surfaceDiffusionCoefficient * params.coverageTimeStep;
    std::vector<NumericType> diagonal(coverage.size(), NumericType(1.));
    bool hasActiveEdge = false;
    for (size_t i = 0; i < coverage.size(); ++i) {
      if (!active[i]) {
        continue;
      }
      for (const auto &edge : diffusionGraph_[i]) {
        if (active[edge.index]) {
          diagonal[i] += diffusionStep * edge.weight;
          hasActiveEdge = true;
        }
      }
    }
    if (!hasActiveEdge) {
      return;
    }

    const auto rhs = coverage;
    auto x = coverage;
    std::vector<NumericType> matrixTimesX, residual(coverage.size(), 0.),
        preconditionedResidual(coverage.size(), 0.),
        direction(coverage.size(), 0.),
        matrixTimesDirection(coverage.size(), 0.);

    applyDiffusionMatrix(x, matrixTimesX, active, diffusionStep);
#pragma omp parallel for
    for (size_t i = 0; i < coverage.size(); ++i) {
      if (active[i]) {
        residual[i] = rhs[i] - matrixTimesX[i];
        preconditionedResidual[i] = residual[i] / diagonal[i];
        direction[i] = preconditionedResidual[i];
      }
    }

    NumericType rz = dotActive(residual, preconditionedResidual, active);
    const NumericType rhsNorm2 =
        std::max(dotActive(rhs, rhs, active), NumericType(1.));
    const NumericType tolerance2 = params.surfaceDiffusionSolverTolerance *
                                   params.surfaceDiffusionSolverTolerance *
                                   rhsNorm2;

    for (unsigned iteration = 0;
         iteration < params.surfaceDiffusionMaxIterations && rz > tolerance2;
         ++iteration) {
      applyDiffusionMatrix(direction, matrixTimesDirection, active,
                           diffusionStep);
      const NumericType denominator =
          dotActive(direction, matrixTimesDirection, active);
      if (std::abs(denominator) <=
          std::numeric_limits<NumericType>::epsilon()) {
        break;
      }

      const NumericType alpha = rz / denominator;
#pragma omp parallel for
      for (size_t i = 0; i < coverage.size(); ++i) {
        if (active[i]) {
          x[i] += alpha * direction[i];
          residual[i] -= alpha * matrixTimesDirection[i];
          preconditionedResidual[i] = residual[i] / diagonal[i];
        }
      }

      const NumericType nextRz =
          dotActive(residual, preconditionedResidual, active);
      if (nextRz <= tolerance2) {
        rz = nextRz;
        break;
      }
      const NumericType beta = nextRz / rz;
#pragma omp parallel for
      for (size_t i = 0; i < coverage.size(); ++i) {
        if (active[i]) {
          direction[i] = preconditionedResidual[i] + beta * direction[i];
        }
      }
      rz = nextRz;
    }

#pragma omp parallel for
    for (size_t i = 0; i < coverage.size(); ++i) {
      if (active[i]) {
        coverage[i] = std::clamp(x[i], NumericType(0.), NumericType(1.));
      }
    }
  }

  const NeutralTransportParameters<NumericType> &params;
  std::vector<Vec3D<NumericType>> cachedCoordinates_{};
  bool hasCoordinates_ = false;
  mutable std::vector<std::vector<DiffusionEdge>> diffusionGraph_{};
  mutable NumericType diffusionNeighborDistance_ = 0.;
  mutable bool diffusionGraphValid_ = false;
};

template <typename NumericType, int D>
class NeutralTransportParticle
    : public viennaray::Particle<NeutralTransportParticle<NumericType, D>,
                                 NumericType> {
public:
  explicit NeutralTransportParticle(
      const NeutralTransportParameters<NumericType> &pParams)
      : params(pParams) {}

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
                    const Vec3D<NumericType> &geomNormal,
                    const unsigned int primID, const int materialId,
                    const viennaray::TracingData<NumericType> *globalData,
                    RNG &rngState) override final {
    NumericType sticking = params.etchFrontSticking;
    if (!MaterialMap::isMaterial(materialId, params.etchFrontMaterial)) {
      const auto theta = globalData == nullptr
                             ? NumericType(0.)
                             : globalData->getVectorData(0)[primID];
      sticking = params.zeroCoverageSticking * (1. - theta);
    }

    sticking = std::clamp(sticking, NumericType(0.), NumericType(1.));
    auto direction =
        viennaray::ReflectionDiffuse<NumericType, D>(geomNormal, rngState);

    return std::pair<NumericType, Vec3D<NumericType>>{sticking, direction};
  }

  NumericType getSourceDistributionPower() const override final {
    return params.sourceDistributionPower;
  }

  std::vector<std::string> getLocalDataLabels() const override final {
    return {params.fluxLabel};
  }

private:
  const NeutralTransportParameters<NumericType> &params;
};

} // namespace impl

#ifdef VIENNACORE_COMPILE_GPU
namespace gpu {

template <typename NumericType, int D>
class NeutralTransport final : public ProcessModelGPU<NumericType, D> {
public:
  explicit NeutralTransport(
      const NeutralTransportParameters<NumericType> &pParams)
      : params(pParams), deviceParams(pParams) {
    initializeModel();
  }

  ~NeutralTransport() override { this->processData.free(); }

private:
  void initializeModel() {
    viennaray::gpu::Particle<NumericType> particle{
        .name = "NeutralTransport",
        .sticking = 0.f, // callable owns all sticking logic
        .cosineExponent = static_cast<float>(params.sourceDistributionPower)};
    particle.dataLabels.push_back(params.fluxLabel);

    std::unordered_map<std::string, unsigned> pMap = {{"NeutralTransport", 0}};
    std::vector<viennaray::gpu::CallableConfig> cMap = {
        {0, viennaray::gpu::CallableSlot::COLLISION,
         "__direct_callable__neutralTransportCollision"},
        {0, viennaray::gpu::CallableSlot::REFLECTION,
         "__direct_callable__neutralTransportReflection"}};
    this->setParticleCallableMap(pMap, cMap);

    auto surfModel =
        SmartPointer<::viennaps::impl::NeutralTransportSurfaceModel<
            NumericType, D>>::New(params);
    auto velField =
        SmartPointer<::viennaps::DefaultVelocityField<NumericType, D>>::New();

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->insertNextParticleType(particle);
    this->setProcessName("NeutralTransport");
    this->setUseMaterialIds(true);
    this->hasGPU = true;

    this->processData.alloc(sizeof(NeutralTransportParametersGPU));
    this->processData.upload(&deviceParams, 1);

    this->processMetaData["Incoming Flux"] = {params.incomingFlux};
    this->processMetaData["Source Pressure"] = {params.sourcePressure};
    this->processMetaData["Source Temperature"] = {params.sourceTemperature};
    this->processMetaData["Source Molecular Mass"] = {
        params.sourceMolecularMass};
    this->processMetaData["Zero Coverage Sticking"] = {
        params.zeroCoverageSticking};
    this->processMetaData["Etch Front Sticking"] = {params.etchFrontSticking};
    this->processMetaData["Desorption Rate"] = {params.desorptionRate};
    this->processMetaData["Desorption Material"] = {
        static_cast<double>(params.desorptionMaterial.legacyId())};
    this->processMetaData["kEtch"] = {params.kEtch};
    this->processMetaData["Surface Site Density"] = {params.surfaceSiteDensity};
    this->processMetaData["Silicon Density"] = {params.siliconDensity};
    this->processMetaData["Coverage Time Step"] = {params.coverageTimeStep};
    this->processMetaData["Source Exponent"] = {params.sourceDistributionPower};
  }

  NeutralTransportParameters<NumericType> params;
  NeutralTransportParametersGPU deviceParams;
};

} // namespace gpu
#endif

template <typename NumericType, int D>
class NeutralTransport : public ProcessModelCPU<NumericType, D> {
public:
  NeutralTransport() { initializeModel(); }

  explicit NeutralTransport(
      const NeutralTransportParameters<NumericType> &pParams)
      : params(pParams) {
    initializeModel();
  }

  void setParameters(const NeutralTransportParameters<NumericType> &pParams) {
    params = pParams;
    initializeModel();
  }

  NeutralTransportParameters<NumericType> &getParameters() { return params; }

#ifdef VIENNACORE_COMPILE_GPU
  SmartPointer<ProcessModelBase<NumericType, D>> getGPUModel() final {
    auto model =
        SmartPointer<gpu::NeutralTransport<NumericType, D>>::New(params);
    model->setProcessName(this->getProcessName().value_or("NeutralTransport"));
    return model;
  }
#endif

private:
  void initializeModel() {
    auto particle =
        std::make_unique<impl::NeutralTransportParticle<NumericType, D>>(
            params);

    auto surfaceModel =
        SmartPointer<impl::NeutralTransportSurfaceModel<NumericType, D>>::New(
            params);

    auto velocityField =
        SmartPointer<DefaultVelocityField<NumericType, D>>::New();

    this->setSurfaceModel(surfaceModel);
    this->setVelocityField(velocityField);
    this->setProcessName("NeutralTransport");
    this->particles.clear();
    this->particleLogSize.clear();
    this->insertNextParticleType(particle);

    this->hasGPU = true;

    this->processMetaData.clear();
    this->processMetaData["Incoming Flux"] = {params.incomingFlux};
    this->processMetaData["Source Pressure"] = {params.sourcePressure};
    this->processMetaData["Source Temperature"] = {params.sourceTemperature};
    this->processMetaData["Source Molecular Mass"] = {
        params.sourceMolecularMass};
    this->processMetaData["Zero Coverage Sticking"] = {
        params.zeroCoverageSticking};
    this->processMetaData["Etch Front Sticking"] = {params.etchFrontSticking};
    this->processMetaData["Desorption Rate"] = {params.desorptionRate};
    this->processMetaData["Desorption Material"] = {
        static_cast<double>(params.desorptionMaterial.legacyId())};
    this->processMetaData["kEtch"] = {params.kEtch};
    this->processMetaData["Surface Site Density"] = {params.surfaceSiteDensity};
    this->processMetaData["Silicon Density"] = {params.siliconDensity};
    this->processMetaData["Coverage Time Step"] = {params.coverageTimeStep};
    this->processMetaData["Surface Diffusion Coefficient"] = {
        params.surfaceDiffusionCoefficient};
    this->processMetaData["Surface Diffusion Material"] = {
        static_cast<double>(params.surfaceDiffusionMaterial.legacyId())};
    this->processMetaData["Surface Diffusion Neighbor Distance"] = {
        params.surfaceDiffusionNeighborDistance};
    this->processMetaData["Surface Diffusion Solver Tolerance"] = {
        params.surfaceDiffusionSolverTolerance};
    this->processMetaData["Surface Diffusion Max Iterations"] = {
        static_cast<double>(params.surfaceDiffusionMaxIterations)};
    this->processMetaData["Source Exponent"] = {params.sourceDistributionPower};
  }

  NeutralTransportParameters<NumericType> params;
};

PS_PRECOMPILE_PRECISION_DIMENSION(NeutralTransport)

} // namespace viennaps
