#pragma once

#include <vcContext.hpp>
#include <vcVectorType.hpp>

#include <raygLaunchParams.hpp>
#include <raygReflection.hpp>

#include <psgPipelineParameters.hpp>

extern "C" __constant__ viennaray::gpu::LaunchParams launchParams;

//
// --- Ion particle
//

__forceinline__ __device__ void
faradayIonCollision(const void *sbtData, viennaray::gpu::PerRayData *prd) {
  viennaps::gpu::impl::IonParams *params =
      (viennaps::gpu::impl::IonParams *)launchParams.customData;
  const bool yieldDefined = abs(params->aSum) > 0.f;

  for (int i = 0; i < prd->ISCount; ++i) {
    auto geomNormal = viennaray::gpu::getNormal(sbtData, prd->primIDs[i]);
    auto cosTheta = __saturatef(
        -viennacore::DotProduct(prd->dir, geomNormal)); // clamp to [0,1]

    float yield = 1.f;
    if (yieldDefined) {
      float cosTheta2 = cosTheta * cosTheta;
      yield = (params->a1 * cosTheta + params->a2 * cosTheta2 +
               params->a3 * cosTheta2 * cosTheta +
               params->a4 * cosTheta2 * cosTheta2) /
              params->aSum;
    }

    // threshold energy is in sqrt scale
    yield *= max(sqrtf(prd->energy) - params->thresholdEnergy, 0.f);

    // In the Faraday cage pipeline, all particle write to the same result array

    // flux array
    atomicAdd(&launchParams.resultBuffer[prd->primIDs[i]],
              prd->rayWeight * yield);

    if (params->redepositionRate > 0.f) {
      // redeposition array
      atomicAdd(&launchParams
                     .resultBuffer[launchParams.numElements + prd->primIDs[i]],
                prd->load);
    }
  }
}
