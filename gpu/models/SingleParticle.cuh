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
singleNeutralCollision(viennaray::gpu::PerRayData *prd) {
  for (int i = 0; i < prd->ISCount; ++i) {
    atomicAdd(&launchParams
                   .resultBuffer[viennaray::gpu::getIdxOffset(0, launchParams) +
                                 prd->primIDs[i]],
              (viennaray::gpu::ResultType)prd->rayWeight);
  }
}

__forceinline__ __device__ void
singleNeutralReflection(const void *sbtData, viennaray::gpu::PerRayData *prd) {
  prd->rayWeight -= prd->rayWeight * launchParams.sticking;
  auto geoNormal = viennaray::gpu::getNormal(sbtData, prd->primID);
  viennaray::gpu::diffuseReflection(prd, geoNormal);
}