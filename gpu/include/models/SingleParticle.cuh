#pragma once

#include <vcContext.hpp>
#include <vcVectorType.hpp>

#include <raygLaunchParams.hpp>
#include <raygReflection.hpp>

extern "C" __constant__ viennaray::gpu::LaunchParams launchParams;

//
// --- Neutral particle
//

__forceinline__ __device__ void
singleNeutralCollision(viennaray::gpu::PerRayData *prd) {
  for (int i = 0; i < prd->ISCount; ++i) {
    atomicAdd(&launchParams.resultBuffer[getIdxOffset(0, launchParams) +
                                         prd->primIDs[i]],
              prd->rayWeight);
  }
}

__forceinline__ __device__ void
singleNeutralReflection(const void *sbtData, viennaray::gpu::PerRayData *prd) {
  prd->rayWeight -= prd->rayWeight * launchParams.sticking;
  auto geoNormal = computeNormal(sbtData, prd->primID);
  diffuseReflection(prd, geoNormal, launchParams.D);
}