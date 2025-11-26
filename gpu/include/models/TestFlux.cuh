#pragma once

#include <vcContext.hpp>
#include <vcVectorType.hpp>

#include <raygLaunchParams.hpp>
#include <raygReflection.hpp>

extern "C" __constant__ viennaray::gpu::LaunchParams launchParams;

//
// --- TestFlux particle
//

__forceinline__ __device__ void
testFluxCollision(viennaray::gpu::PerRayData *prd) {
  for (int i = 0; i < prd->ISCount; ++i) {
    if (prd->numReflections == 0) {
      atomicAdd(
          &launchParams
               .resultBuffer[viennaray::gpu::getIdxOffset(0, launchParams) +
                             prd->primIDs[i]],
          prd->rayWeight);
    } else {
      atomicAdd(
          &launchParams
               .resultBuffer[viennaray::gpu::getIdxOffset(1, launchParams) +
                             prd->primIDs[i]],
          prd->rayWeight);
    }
  }
}