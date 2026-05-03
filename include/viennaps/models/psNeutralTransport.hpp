#pragma once

#include "../materials/psMaterialMap.hpp"
#include "../process/psProcessModel.hpp"
#include "../psUnits.hpp"

#include <rayParticle.hpp>
#include <rayReflection.hpp>

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

template <typename NumericType> constexpr NumericType avogadroNumber() {
  return static_cast<NumericType>(6.02214076e23);
}

template <typename NumericType> constexpr NumericType boltzmannConstant() {
  return static_cast<NumericType>(1.380649e-23);
}

template <typename NumericType> constexpr NumericType atomicMassUnit() {
  return static_cast<NumericType>(1.66053906660e-27);
}

template <typename NumericType>
NumericType molecularEffusionFlux(NumericType pressurePa,
                                  NumericType temperatureK,
                                  NumericType molecularMassAmu) {
  if (pressurePa <= NumericType(0.) || temperatureK <= NumericType(0.) ||
      molecularMassAmu <= NumericType(0.)) {
    return NumericType(0.);
  }

  const auto molecularMassKg = molecularMassAmu * atomicMassUnit<NumericType>();
  return pressurePa /
         std::sqrt(NumericType(2.) * NumericType(M_PI) * molecularMassKg *
                   boltzmannConstant<NumericType>() * temperatureK);
}

template <typename NumericType> struct NeutralTransportParameters {
  // Gas-phase incoming neutral flux scale in molecules / (m^2 s). The ray
  // tracer provides the local dimensionless transport factor.
  NumericType incomingFlux = 1.;

  // Fixed-pressure reservoir boundary condition used by the benchmark paper.
  NumericType sourcePressure = 1.;          // Pa
  NumericType sourceTemperature = 300.;     // K
  NumericType sourceMolecularMass = 18.998; // amu, atomic fluorine

  // Langmuir adsorption/desorption parameters.
  NumericType zeroCoverageSticking = 0.1;
  NumericType etchFrontSticking = 1.;
  NumericType desorptionRate = 0.;          // 1 / s
  Material desorptionMaterial = Material::Mask;
  NumericType kEtch = 0.;                   // 1 / s
  NumericType surfaceSiteDensity = 1.66e-5; // mol / m^2
  NumericType siliconDensity = 8.3e4;       // mol / m^3
  NumericType coverageTimeStep = 1.;        // s
  bool useSteadyStateCoverage = true;

  // Placeholder for Eq. (9) in the neutral transport paper. The base skeleton
  // applies a 1D benchmark-only diffusion along a cylinder sidewall band.
  NumericType surfaceDiffusionCoefficient = 0.;
  NumericType surfaceDiffusionRadius = 0.;
  NumericType surfaceDiffusionTolerance = 0.;

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
        velocity->at(i) =
            -etchVelocity * units::Time::convertSecond() /
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
              ? sticking * params.incomingFlux *
                    neutralFlux->at(i) /
                    (avogadroNumber<NumericType>() * params.surfaceSiteDensity)
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

    if (params.surfaceDiffusionCoefficient > 0. &&
        params.surfaceDiffusionRadius > 0. &&
        params.surfaceDiffusionTolerance > 0. && hasCoordinates_) {
      applyBenchmarkDiffusion(cachedCoordinates_, *coverage);
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
                   avogadroNumber<NumericType>() / params.incomingFlux;
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
  static NumericType pointRadius(const Vec3D<NumericType> &point) {
    if constexpr (D == 2) {
      return std::abs(point[0]);
    } else {
      NumericType sum = 0.;
      for (int i = 0; i < D - 1; ++i) {
        sum += point[i] * point[i];
      }
      return std::sqrt(sum);
    }
  }

  void
  applyBenchmarkDiffusion(const std::vector<Vec3D<NumericType>> &coordinates,
                          std::vector<NumericType> &coverage) const {
    if (coordinates.size() != coverage.size()) {
      return;
    }

    struct Candidate {
      size_t index;
      NumericType z;
    };

    std::vector<Candidate> candidates;
    candidates.reserve(coordinates.size());

#ifdef _OPENMP
    const int numThreads = omp_get_max_threads();
    std::vector<std::vector<Candidate>> threadLocalCandidates(numThreads);

#pragma omp parallel
    {
      const int threadId = omp_get_thread_num();
      auto &localCandidates = threadLocalCandidates[threadId];

#pragma omp for nowait
      for (size_t i = 0; i < coordinates.size(); ++i) {
        const auto radius = pointRadius(coordinates[i]);
        if (std::abs(radius - params.surfaceDiffusionRadius) <=
            params.surfaceDiffusionTolerance) {
          localCandidates.push_back(Candidate{i, coordinates[i][D - 1]});
        }
      }
    }

    for (auto &localCandidates : threadLocalCandidates) {
      candidates.insert(candidates.end(), localCandidates.begin(),
                        localCandidates.end());
    }
#else
    for (size_t i = 0; i < coordinates.size(); ++i) {
      const auto radius = pointRadius(coordinates[i]);
      if (std::abs(radius - params.surfaceDiffusionRadius) <=
          params.surfaceDiffusionTolerance) {
        candidates.push_back(Candidate{i, coordinates[i][D - 1]});
      }
    }
#endif

    if (candidates.size() < 2) {
      return;
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate &a, const Candidate &b) {
                if (a.z != b.z)
                  return a.z < b.z;
                return a.index < b.index;
              });

    NumericType avgSpacing = 0.;
    for (size_t i = 1; i < candidates.size(); ++i) {
      avgSpacing += std::abs(candidates[i].z - candidates[i - 1].z);
    }
    avgSpacing /= static_cast<NumericType>(candidates.size() - 1);
    if (avgSpacing <= std::numeric_limits<NumericType>::epsilon()) {
      return;
    }

    const NumericType r = params.surfaceDiffusionCoefficient *
                          params.coverageTimeStep / (avgSpacing * avgSpacing);
    if (r <= NumericType(0.)) {
      return;
    }

    const size_t n = candidates.size();
    std::vector<NumericType> a(n, -r), b(n, 1. + 2. * r), c(n, -r), d(n, 0.),
        x(n, 0.);

    // Neumann zero-gradient boundary conditions via mirrored ghost points.
    c[0] = -2. * r;
    a[n - 1] = -2. * r;

    for (size_t i = 0; i < n; ++i) {
      d[i] = coverage[candidates[i].index];
    }

    // Thomas algorithm for the tridiagonal solve.
    for (size_t i = 1; i < n; ++i) {
      const NumericType m = a[i] / b[i - 1];
      b[i] -= m * c[i - 1];
      d[i] -= m * d[i - 1];
    }

    x[n - 1] = d[n - 1] / b[n - 1];
    for (size_t i = n - 1; i-- > 0;) {
      x[i] = (d[i] - c[i] * x[i + 1]) / b[i];
    }

    for (size_t i = 0; i < n; ++i) {
      coverage[candidates[i].index] =
          std::clamp(x[i], NumericType(0.), NumericType(1.));
    }
  }

  const NeutralTransportParameters<NumericType> &params;
  std::vector<Vec3D<NumericType>> cachedCoordinates_{};
  bool hasCoordinates_ = false;
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
        .cosineExponent =
            static_cast<float>(params.sourceDistributionPower)};
    particle.dataLabels.push_back(params.fluxLabel);

    std::unordered_map<std::string, unsigned> pMap = {
        {"NeutralTransport", 0}};
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
    model->setProcessName(
        this->getProcessName().value_or("NeutralTransport"));
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
    this->processMetaData["Surface Diffusion Radius"] = {
        params.surfaceDiffusionRadius};
    this->processMetaData["Surface Diffusion Tolerance"] = {
        params.surfaceDiffusionTolerance};
    this->processMetaData["Source Exponent"] = {params.sourceDistributionPower};
  }

  NeutralTransportParameters<NumericType> params;
};

PS_PRECOMPILE_PRECISION_DIMENSION(NeutralTransport)

} // namespace viennaps
