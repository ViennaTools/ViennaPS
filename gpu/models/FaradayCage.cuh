#pragma once

#include "vcContext.hpp"
#include "vcVectorType.hpp"

#include "raygLaunchParams.hpp"
#include "raygReflection.hpp"

#include "models/psIonBeamParameters.hpp"

extern "C" __constant__ viennaray::gpu::LaunchParams launchParams;

//
// --- Ion particle
//

__forceinline__ __device__ void
faradayIonCollision(const void *sbtData, viennaray::gpu::PerRayData *prd,
                    unsigned int primID) {
  viennaps::gpu::IonParams *params =
      (viennaps::gpu::IonParams *)launchParams.customData;
  const bool yieldDefined = abs(params->aSum) > 0.f;

  auto geomNormal = viennaray::gpu::getNormal(sbtData, primID);
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
  atomicAdd(&launchParams.resultBuffer[primID],
            (viennaray::gpu::ResultType)prd->rayWeight * yield);

  if (params->redepositionRate > 0.f) {
    // redeposition array
    atomicAdd(&launchParams.resultBuffer[launchParams.numElements + primID],
              (viennaray::gpu::ResultType)prd->load);
  }
}
