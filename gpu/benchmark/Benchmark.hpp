#pragma once

#include <geometries/psMakeHole.hpp>
#include <geometries/psMakeTrench.hpp>
#include <models/psIonBeamEtching.hpp>
#include <psDomain.hpp>
#include <rayParticle.hpp>

#define MAKE_GEO Hole
#define DEFAULT_GRID_DELTA 0.1
#define DEFAULT_STICKING 0.1
#define DIM 3

constexpr int particleType = 0;
using TranslatorType = std::unordered_map<unsigned long, unsigned long>;

template <class NumericType>
auto Trench(NumericType gridDelta = DEFAULT_GRID_DELTA) {
  NumericType xExtent = 10.;
  NumericType yExtent = 10.;
  NumericType width = 5.;
  NumericType depth = 25.;

  using namespace viennaps;
  auto domain = Domain<NumericType, DIM>::New(
      gridDelta, xExtent, yExtent, BoundaryType::REFLECTIVE_BOUNDARY);
  MakeTrench<NumericType, DIM>(domain, width, depth).apply().apply();
  return domain;
}

template <class NumericType>
auto Hole(NumericType gridDelta = DEFAULT_GRID_DELTA) {
  NumericType xExtent = 10.;
  NumericType yExtent = 10.;
  NumericType radius = 3.0;
  NumericType depth = 30.;

  using namespace viennaps;
  auto domain = Domain<NumericType, DIM>::New(
      gridDelta, xExtent, yExtent, BoundaryType::REFLECTIVE_BOUNDARY);
  MakeHole<NumericType, DIM>(domain, radius, depth).apply();
  return domain;
}

template <class NumericType, int N>
std::array<NumericType, N> linspace(NumericType start, NumericType end) {
  std::array<NumericType, N> arr{};
  NumericType step = (end - start) / static_cast<NumericType>(N - 1);
  for (int i = 0; i < N; ++i) {
    arr[i] = start + i * step;
  }
  return arr;
}

template <typename NumericType> auto getIBEParameters() {
  viennaps::IBEParameters<NumericType> params;
  params.redepositionRate = 1.0;
  params.cos4Yield.isDefined = true;
  params.cos4Yield.a1 = 0.1;
  params.cos4Yield.a2 = 0.2;
  params.cos4Yield.a3 = 0.3;
  params.cos4Yield.a4 = 0.4;
  params.thetaRMin = 0.0;
  params.thetaRMax = 90.0;
  params.meanEnergy = 250.0;
  params.sigmaEnergy = 10.0;
  params.thresholdEnergy = 20.0;
  params.n_l = 10;
  params.inflectAngle = 80.0;
  params.minAngle = 85.0;
  params.exponent = 100.0;
  return params;
}

template <typename NumericType, int D> auto makeCPUParticle() {
  if constexpr (particleType == 0) {
    auto particle =
        std::make_unique<viennaray::DiffuseParticle<NumericType, D>>(
            DEFAULT_STICKING, "flux");
    return particle;
  } else if constexpr (particleType == 1) {
    auto params = getIBEParameters<NumericType>();
    auto particle = std::make_unique<
        viennaps::impl::IBEIonWithRedeposition<NumericType, D>>(params);
    return particle;
  } else {
    static_assert(particleType == 0 || particleType == 1,
                  "Unsupported particle type!");
  }
}

template <typename NumericType, int D>
std::tuple<viennaray::gpu::Particle<NumericType>,
           std::unordered_map<std::string, unsigned>,
           std::vector<viennaray::gpu::CallableConfig>>
makeGPUParticle() {
  if constexpr (particleType == 0) {
    auto particle = viennaray::gpu::Particle<NumericType>();
    particle.name = "SingleParticle";
    particle.sticking = DEFAULT_STICKING;
    particle.dataLabels.emplace_back("flux");
    std::unordered_map<std::string, unsigned> pMap = {{"SingleParticle", 0}};
    std::vector<viennaray::gpu::CallableConfig> cMap = {
        {0, viennaray::gpu::CallableSlot::COLLISION,
         "__direct_callable__singleNeutralCollision"},
        {0, viennaray::gpu::CallableSlot::REFLECTION,
         "__direct_callable__singleNeutralReflection"}};
    return {particle, pMap, cMap};
  } else if constexpr (particleType == 1) {
    auto params_ = getIBEParameters<NumericType>();
    viennaray::gpu::Particle<NumericType> particle{
        .name = "IBEIon", .cosineExponent = params_.exponent};
    particle.dataLabels.push_back("flux");
    if (params_.redepositionRate > 0.) {
      particle.dataLabels.push_back(
          viennaps::impl::IBESurfaceModel<NumericType>::redepositionLabel);
    }
    std::unordered_map<std::string, unsigned> pMap = {{"IBEIon", 0}};
    std::vector<viennaray::gpu::CallableConfig> cMap = {
        {0, viennaray::gpu::CallableSlot::COLLISION,
         "__direct_callable__IBECollision"},
        {0, viennaray::gpu::CallableSlot::REFLECTION,
         "__direct_callable__IBEReflection"},
        {0, viennaray::gpu::CallableSlot::INIT, "__direct_callable__IBEInit"}};
    return {particle, pMap, cMap};
  } else {
    static_assert(particleType == 0 || particleType == 1,
                  "Unsupported particle type!");
  }
}

auto getDeviceParams() {
  if constexpr (particleType != 1) {
    throw std::runtime_error(
        "getDeviceParams is only defined for particleType 1 (IBEIon)!");
  }
  auto params_ = getIBEParameters<float>();
  // Parameters to upload to device
  viennaps::gpu::impl::IonParams deviceParams;
  deviceParams.thetaRMin = viennaps::constants::degToRad(params_.thetaRMin);
  deviceParams.thetaRMax = viennaps::constants::degToRad(params_.thetaRMax);
  deviceParams.meanEnergy = params_.meanEnergy;
  deviceParams.sigmaEnergy = params_.sigmaEnergy;
  deviceParams.thresholdEnergy =
      std::sqrt(params_.thresholdEnergy); // precompute sqrt
  deviceParams.minAngle = viennaps::constants::degToRad(params_.minAngle);
  deviceParams.inflectAngle =
      viennaps::constants::degToRad(params_.inflectAngle);
  deviceParams.n_l = params_.n_l;
  deviceParams.B_sp = 0.f; // not used in IBE
  if (params_.cos4Yield.isDefined) {
    deviceParams.a1 = params_.cos4Yield.a1;
    deviceParams.a2 = params_.cos4Yield.a2;
    deviceParams.a3 = params_.cos4Yield.a3;
    deviceParams.a4 = params_.cos4Yield.a4;
    deviceParams.aSum = params_.cos4Yield.aSum();
  }
  deviceParams.redepositionRate = params_.redepositionRate;
  deviceParams.redepositionThreshold = params_.redepositionThreshold;

  return deviceParams;
}