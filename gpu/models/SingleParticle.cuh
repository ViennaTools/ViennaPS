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
singleNeutralCollision(viennaray::gpu::PerRayData *prd, unsigned int primID) {
  atomicAdd(
      &launchParams
           .resultBuffer[viennaray::gpu::getIdxOffset(0, launchParams, primID)],
      (viennaray::gpu::ResultType)prd->rayWeight);
}

__forceinline__ __device__ void
singleNeutralReflection(const void *sbtData, viennaray::gpu::PerRayData *prd,
                        unsigned int primID) {
  prd->rayWeight -= prd->rayWeight * launchParams.sticking;
  auto geoNormal = viennaray::gpu::getNormal(sbtData, primID);
  viennaray::gpu::diffuseReflection(prd, geoNormal);
}