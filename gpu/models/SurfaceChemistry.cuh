#pragma once

#include "vcContext.hpp"
#include "vcVectorType.hpp"

#include "raygLaunchParams.hpp"
#include "raygReflection.hpp"

#include "models/psIonModelUtil.hpp"

extern "C" __constant__ viennaray::gpu::LaunchParams launchParams;

namespace viennaps {

// Device-side data for the generic chemical-deposition particles.
//
// A neutral is absorbed with the probability that it reacts at the hit, the
// sum over the reactions that consume it of their rate laws per unit incident
// flux. Those reactions are indexed by launchParams.particleIdx, because a
// mechanism may trace several gas species that react in different ways.
struct SurfaceChemistryParamsGPU {
  static constexpr int maxParticles = 16; // a published mechanism
                                          // can adsorb a dozen species
  static constexpr int maxCoverages = 16;
  static constexpr int maxMaterials = 8; // per-channel rate overrides
  static constexpr int maxChannels = 4;  // reactions consuming one species
  static constexpr int maxSiteTypes = 4;
  static constexpr int maxChannelFactors = 3; // coverage factors per reaction

  int numCoverages = 0;
  int numSiteTypes = 1;
  int coverageSite[maxCoverages] = {}; // site-type index of each coverage

  // The reactions that consume each traced species: the rate constant at the
  // mechanism temperature, which `channelK` gives unless the material under
  // the hit is listed, and the free-site and coverage exponents.
  int numChannels[maxParticles] = {};
  float channelK[maxParticles][maxChannels] = {};
  int channelNumOverrides[maxParticles][maxChannels] = {};
  int channelOverrideMaterial[maxParticles][maxChannels][maxMaterials] =
      {}; // legacy material ids
  float channelOverrideK[maxParticles][maxChannels][maxMaterials] = {};
  int channelFreeExp[maxParticles][maxChannels][maxSiteTypes] = {};
  int channelNumFactors[maxParticles][maxChannels] = {};
  int channelFactorIndex[maxParticles][maxChannels][maxChannelFactors] = {};
  int channelFactorExp[maxParticles][maxChannels][maxChannelFactors] = {};

  // --- ions -----------------------------------------------------------------
  // Everything below comes from the reaction file, by way of the mechanism.
  static constexpr int maxYields = 6;

  float meanEnergy = 100.f, sigmaEnergy = 10.f;
  float inflectAngle = 89.f, n_l = 10.f; // radians once uploaded
  float minAngle = 80.f, thetaRMin = 70.f, thetaRMax = 90.f;
  float minEth = 0.f; // stop the ion once it can drive nothing

  int numYields = 0;
  float yieldA[maxYields] = {};
  float yieldEth[maxYields] = {};
  float yieldB[maxYields] = {};
  int yieldEnhanced[maxYields] = {};
  // per-material A and Eth, so a mask is harder to sputter
  int yieldNumOverrides[maxYields] = {};
  int yieldOverrideMaterial[maxYields][maxMaterials] = {};
  float yieldOverrideA[maxYields][maxMaterials] = {};
  float yieldOverrideEth[maxYields][maxMaterials] = {};
};

// The host and the device each carry their own copy of this struct, and a
// difference between them is silent: the device reads the right bytes at the
// wrong offsets, so a coverage's site index becomes garbage and the chemistry
// quietly changes. Both copies assert the same shape, so editing one without
// the other fails the build instead.
static_assert(SurfaceChemistryParamsGPU::maxParticles == 16 &&
                  SurfaceChemistryParamsGPU::maxCoverages == 16 &&
                  SurfaceChemistryParamsGPU::maxMaterials == 8 &&
                  SurfaceChemistryParamsGPU::maxChannels == 4 &&
                  SurfaceChemistryParamsGPU::maxSiteTypes == 4 &&
                  SurfaceChemistryParamsGPU::maxChannelFactors == 3 &&
                  SurfaceChemistryParamsGPU::maxYields == 6,
              "SurfaceChemistryParamsGPU must have the same shape in "
              "psSurfaceChemistry.hpp and gpu/models/SurfaceChemistry.cuh");

} // namespace viennaps

// theta_*t = 1 - sum_{i in t} theta_i for site type t, read from the
// per-element coverage buffer. Coverage i of element e sits at
// e + i * numElements, the layout the surface model uploads.
__forceinline__ __device__ float
chemicalFreeSiteFraction(const float *coverages,
                         const viennaps::SurfaceChemistryParamsGPU *params,
                         const int site, const unsigned primID) {
  float occupied = 0.f;
  for (int i = 0; i < params->numCoverages; ++i)
    if (params->coverageSite[i] == site)
      occupied += coverages[primID + i * launchParams.numElements];
  return fmaxf(1.f - occupied, 0.f);
}

// The particle records the RAW incident flux. The sticking is applied once in
// the rate law and once in the re-emission below; applying it here as well
// would count it twice.
__forceinline__ __device__ void
chemicalNeutralCollision(const void *, viennaray::gpu::PerRayData *prd,
                         unsigned int primID) {
  atomicAdd(&launchParams
                 .resultBuffer[viennaray::gpu::getIdxOffset(0, launchParams) +
                               primID],
            (viennaray::gpu::ResultType)prd->rayWeight);
}

// The rate constant of reaction c of this particle on the material under the
// hit. Mirrors the per-material lookup the CPU particle does, so the two
// engines agree.
__forceinline__ __device__ float
chemicalChannelRate(const viennaps::SurfaceChemistryParamsGPU *params,
                    const int p, const int c, const unsigned primID) {
  const int count = params->channelNumOverrides[p][c];
  if (count > 0) {
    const int consecutiveId = launchParams.materialIds[primID];
    const int legacyId = launchParams.materialMap[consecutiveId];
    for (int i = 0; i < count; ++i)
      if (params->channelOverrideMaterial[p][c][i] == legacyId)
        return params->channelOverrideK[p][c][i];
  }
  return params->channelK[p][c];
}

// The probability that the particle reacts at the hit, the sum over the
// reactions that consume it of k * prod theta_*t^n * prod theta_i^n, the same
// law the CPU particle uses.
__forceinline__ __device__ void
chemicalNeutralReflection(const void *sbtData, viennaray::gpu::PerRayData *prd,
                          unsigned int primID) {
  const viennaps::SurfaceChemistryParamsGPU *params =
      reinterpret_cast<const viennaps::SurfaceChemistryParamsGPU *>(
          launchParams.customData);
  const viennaray::gpu::HitSBTDataBase *baseData =
      reinterpret_cast<const viennaray::gpu::HitSBTDataBase *>(sbtData);
  const float *coverages = (const float *)baseData->cellData;
  const int p = launchParams.particleIdx;

  float sEff = 0.f;
  for (int c = 0; c < params->numChannels[p]; ++c) {
    float v = chemicalChannelRate(params, p, c, primID);
    for (int t = 0; t < params->numSiteTypes; ++t) {
      const int n = params->channelFreeExp[p][c][t];
      if (n == 0)
        continue;
      const float thetaFree =
          chemicalFreeSiteFraction(coverages, params, t, primID);
      for (int e = 0; e < n; ++e)
        v *= thetaFree;
    }
    for (int f = 0; f < params->channelNumFactors[p][c]; ++f) {
      const float theta =
          coverages[primID + params->channelFactorIndex[p][c][f] *
                                 launchParams.numElements];
      for (int e = 0; e < params->channelFactorExp[p][c][f]; ++e)
        v *= theta;
    }
    sEff += v;
  }

  prd->rayWeight -= prd->rayWeight * __saturatef(sEff);
  auto geoNormal = viennaray::gpu::getNormal(sbtData, primID);
  viennaray::gpu::diffuseReflection(prd, geoNormal);
}

//
// --- the ion
//
// Mirrors impl::ChemicalIon on the CPU: the yield is evaluated here and
// deposited as a flux, so the surface chemistry sees an ordinary flux and the
// solver needs no notion of ions. Every parameter comes from the reaction file.
//

__forceinline__ __device__ const viennaps::SurfaceChemistryParamsGPU *
chemicalParams() {
  return reinterpret_cast<const viennaps::SurfaceChemistryParamsGPU *>(
      launchParams.customData);
}

__forceinline__ __device__ void
chemicalIonInit(viennaray::gpu::PerRayData *prd) {
  const auto *p = chemicalParams();
  viennaps::impl::initNormalDistEnergy(prd, p->meanEnergy, p->sigmaEnergy);
}

__forceinline__ __device__ void
chemicalIonCollision(const void *sbtData, viennaray::gpu::PerRayData *prd,
                     unsigned int primID) {
  const auto *p = chemicalParams();
  const int consecutiveId = launchParams.materialIds[primID];
  const int legacyId = launchParams.materialMap[consecutiveId];

  auto geomNormal = viennaray::gpu::getNormal(sbtData, primID);
  const float cosTheta =
      __saturatef(-viennacore::DotProduct(prd->dir, geomNormal));
  const float angle = acosf(cosTheta);
  const float sqrtE = sqrtf(prd->energy);

  for (int c = 0; c < p->numYields; ++c) {
    float A = p->yieldA[c];
    float Eth = p->yieldEth[c];
    for (int k = 0; k < p->yieldNumOverrides[c]; ++k)
      if (p->yieldOverrideMaterial[c][k] == legacyId) {
        A = p->yieldOverrideA[c][k];
        Eth = p->yieldOverrideEth[c][k];
        break;
      }

    float f;
    if (p->yieldEnhanced[c]) {
      f = cosTheta < 0.5f ? fmaxf(3.f - 6.f * angle / M_PIf, 0.f) : 1.f;
    } else {
      f = fmaxf((1.f + p->yieldB[c] * (1.f - cosTheta * cosTheta)) * cosTheta,
                0.f);
    }

    const float Y = A * fmaxf(sqrtE - sqrtf(Eth), 0.f) * f;
    atomicAdd(
        &launchParams
             .resultBuffer[viennaray::gpu::getIdxOffset(c, launchParams) +
                           primID],
        (viennaray::gpu::ResultType)(Y * prd->rayWeight));
  }
}

__forceinline__ __device__ void
chemicalIonReflection(const void *sbtData, viennaray::gpu::PerRayData *prd,
                      unsigned int primID) {
  const auto *p = chemicalParams();
  auto geomNormal = viennaray::gpu::getNormal(sbtData, primID);
  const float cosTheta =
      __saturatef(-viennacore::DotProduct(prd->dir, geomNormal));
  const float angle = acosf(cosTheta);

  // a steep hit is absorbed, a glancing one reflects
  float sticking = 1.f;
  if (angle > p->thetaRMin)
    sticking = 1.f - __saturatef((angle - p->thetaRMin) /
                                 (p->thetaRMax - p->thetaRMin));
  if (sticking >= 1.f) {
    prd->rayWeight = 0.f;
    return;
  }

  viennaps::impl::updateEnergy(prd, p->inflectAngle, p->n_l, angle);
  if (prd->energy > p->minEth) {
    prd->rayWeight -= prd->rayWeight * sticking;
    viennaray::gpu::conedCosineReflection(prd, geomNormal,
                                          M_PI_2f - fminf(angle, p->minAngle));
  } else {
    prd->rayWeight = 0.f;
  }
}
