#pragma once

#include "vcContext.hpp"
#include "vcVectorType.hpp"

#include "raygLaunchParams.hpp"
#include "raygReflection.hpp"

#include "models/psPipelineParameters.hpp"

extern "C" __constant__ viennaray::gpu::LaunchParams launchParams;

namespace viennaps {

// Device-side data for the generic chemical-deposition particles.
//
// The sticking s(T) already travels per particle in launchParams.sticking, so
// only the free-site exponent and the number of coverages are model specific.
// The exponent is indexed by launchParams.particleIdx, because a mechanism may
// trace several gas species whose adsorption steps take different site counts.
struct SurfaceChemistryParamsGPU {
  static constexpr int maxParticles = 16; // a published mechanism
                                          // can adsorb a dozen species
  static constexpr int maxCoverages = 16;
  static constexpr int maxMaterials = 8; // per-particle sticking overrides

  int numCoverages = 0;
  int coverageSite[maxCoverages] = {}; // site-type index of each coverage
  int freeSiteExponent[maxParticles] = {};
  int stickingSite[maxParticles] = {}; // site type each particle sticks to

  // Sticking per particle, already evaluated at the mechanism temperature.
  // `defaultSticking` applies unless the material under the hit is listed, so a
  // species that adsorbs on one material and not another reflects accordingly.
  float defaultSticking[maxParticles] = {};
  int numOverrides[maxParticles] = {};
  int overrideMaterial[maxParticles][maxMaterials] = {}; // legacy material ids
  float overrideSticking[maxParticles][maxMaterials] = {};

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
                  SurfaceChemistryParamsGPU::maxYields == 6,
              "SurfaceChemistryParamsGPU must have the same shape in "
              "psSurfaceChemistry.hpp and gpu/models/SurfaceChemistry.cuh");

} // namespace viennaps

// theta_*t = 1 - sum_{i in t} theta_i for the site type t this particle sticks
// to, read from the per-element coverage buffer. Coverage i of element e sits at
// e + i * numElements, the layout the surface model uploads.
__forceinline__ __device__ float
chemicalFreeSiteFraction(const void *sbtData, const unsigned primID) {
  const viennaray::gpu::HitSBTDataBase *baseData =
      reinterpret_cast<const viennaray::gpu::HitSBTDataBase *>(sbtData);
  const float *coverages = (const float *)baseData->cellData;
  const viennaps::SurfaceChemistryParamsGPU *params =
      reinterpret_cast<const viennaps::SurfaceChemistryParamsGPU *>(
          launchParams.customData);

  const int site = params->stickingSite[launchParams.particleIdx];
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
chemicalNeutralCollision(const void *, viennaray::gpu::PerRayData *prd) {
  for (int i = 0; i < prd->ISCount; ++i) {
    atomicAdd(&launchParams
                   .resultBuffer[viennaray::gpu::getIdxOffset(0, launchParams) +
                                 prd->primIDs[i]],
              (viennaray::gpu::ResultType)prd->rayWeight);
  }
}

// The sticking of this particle on the material under the hit. Mirrors the
// per-material lookup the CPU particle does, so the two engines agree.
__forceinline__ __device__ float chemicalSticking(const unsigned primID) {
  const viennaps::SurfaceChemistryParamsGPU *params =
      reinterpret_cast<const viennaps::SurfaceChemistryParamsGPU *>(
          launchParams.customData);
  const int p = launchParams.particleIdx;
  const int count = params->numOverrides[p];
  if (count > 0) {
    const int consecutiveId = launchParams.materialIds[primID];
    const int legacyId = launchParams.materialMap[consecutiveId];
    for (int i = 0; i < count; ++i)
      if (params->overrideMaterial[p][i] == legacyId)
        return params->overrideSticking[p][i];
  }
  return params->defaultSticking[p];
}

// s_eff = s(T) * theta_free^n, the same law the CPU particle uses.
__forceinline__ __device__ void
chemicalNeutralReflection(const void *sbtData,
                          viennaray::gpu::PerRayData *prd) {
  const viennaps::SurfaceChemistryParamsGPU *params =
      reinterpret_cast<const viennaps::SurfaceChemistryParamsGPU *>(
          launchParams.customData);
  const float thetaFree = chemicalFreeSiteFraction(sbtData, prd->primID);

  float sEff = chemicalSticking(prd->primID);
  const int n = params->freeSiteExponent[launchParams.particleIdx];
  for (int e = 0; e < n; ++e)
    sEff *= thetaFree;

  prd->rayWeight -= prd->rayWeight * __saturatef(sEff);
  auto geoNormal = viennaray::gpu::getNormal(sbtData, prd->primID);
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
  viennaps::gpu::impl::initNormalDistEnergy(prd, p->meanEnergy, p->sigmaEnergy);
}

__forceinline__ __device__ void
chemicalIonCollision(const void *sbtData, viennaray::gpu::PerRayData *prd) {
  const auto *p = chemicalParams();
  for (int i = 0; i < prd->ISCount; ++i) {
    const int consecutiveId = launchParams.materialIds[prd->primIDs[i]];
    const int legacyId = launchParams.materialMap[consecutiveId];

    auto geomNormal = viennaray::gpu::getNormal(sbtData, prd->primIDs[i]);
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
      atomicAdd(&launchParams.resultBuffer[viennaray::gpu::getIdxOffset(
                    c, launchParams) +
                                           prd->primIDs[i]],
                (viennaray::gpu::ResultType)(Y * prd->rayWeight));
    }
  }
}

__forceinline__ __device__ void
chemicalIonReflection(const void *sbtData, viennaray::gpu::PerRayData *prd) {
  const auto *p = chemicalParams();
  auto geomNormal = viennaray::gpu::getNormal(sbtData, prd->primID);
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

  viennaps::gpu::impl::updateEnergy(prd, p->inflectAngle, p->n_l, angle);
  if (prd->energy > p->minEth) {
    prd->rayWeight -= prd->rayWeight * sticking;
    viennaray::gpu::conedCosineReflection(prd, geomNormal,
                                          M_PI_2f - fminf(angle, p->minAngle));
  } else {
    prd->rayWeight = 0.f;
  }
}
