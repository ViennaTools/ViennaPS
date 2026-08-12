#pragma once

#include "vcContext.hpp"
#include "vcVectorType.hpp"

#include "raygLaunchParams.hpp"
#include "raygReflection.hpp"

extern "C" __constant__ viennaray::gpu::LaunchParams launchParams;

//
// --- Neutral particle
//

__forceinline__ __device__ void
singleALDNeutralReflection(const void *sbtData,
                           viennaray::gpu::PerRayData *prd) {
  const viennaray::gpu::HitSBTDataBase *baseData =
      reinterpret_cast<const viennaray::gpu::HitSBTDataBase *>(sbtData);
  float *data = (float *)(baseData->cellData);
  const float coverage = data[prd->primID];
  const float Seff = launchParams.sticking * max(1.f - coverage, 0.f);
  prd->rayWeight -= prd->rayWeight * Seff;
  auto geoNormal = viennaray::gpu::getNormal(sbtData, prd->primID);
  viennaray::gpu::diffuseReflection(prd, geoNormal);
}

__forceinline__ __device__ void
singleALDNeutralCollision(const void *sbtData,
                          viennaray::gpu::PerRayData *prd) {
  const viennaray::gpu::HitSBTDataBase *baseData =
      reinterpret_cast<const viennaray::gpu::HitSBTDataBase *>(sbtData);
  float *data = (float *)(baseData->cellData);
  const float coverage = data[prd->primID];
  const float Seff = launchParams.sticking * max(1.f - coverage, 0.f);
  float flux = prd->rayWeight * Seff;

  for (int i = 0; i < prd->ISCount; ++i) {
    atomicAdd(&launchParams
                   .resultBuffer[viennaray::gpu::getIdxOffset(0, launchParams) +
                                 prd->primIDs[i]],
              (viennaray::gpu::ResultType)flux);
  }
}