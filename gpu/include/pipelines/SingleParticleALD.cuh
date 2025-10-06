#pragma once

#include <vcContext.hpp>
#include <vcVectorType.hpp>

#include "raygLaunchParams.hpp"

#include <raygReflection.hpp>

extern "C" __constant__ viennaray::gpu::LaunchParams launchParams;

//
// --- Neutral particle
//

__forceinline__ __device__ void
singleALDNeutralCollision(const void *sbtData,
                          viennaray::gpu::PerRayData *prd) {
  const HitSBTDataBase *baseData =
      reinterpret_cast<const HitSBTDataBase *>(sbtData);
  float *data = (float *)(baseData->cellData);
  for (int i = 0; i < prd->ISCount; ++i) {
    const float coverage = data[prd->TIndex[i]];
    const float Seff = launchParams.sticking * max(1.f - coverage, 0.f);
    atomicAdd(&launchParams.resultBuffer[getIdxOffset(0, launchParams) +
                                         prd->TIndex[i]],
              prd->rayWeight * Seff);
  }
}