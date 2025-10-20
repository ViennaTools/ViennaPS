#pragma once

#include <vcContext.hpp>
#include <vcVectorType.hpp>

#include "raygLaunchParams.hpp"
#include <raygReflection.hpp>

#include <models/psgPipelineParameters.hpp>

extern "C" __constant__ viennaray::gpu::LaunchParams launchParams;
using namespace viennaray::gpu;

//
// --- Ion particle
//

__forceinline__ __device__ void
faradayIonCollision(const void *sbtData, viennaray::gpu::PerRayData *prd) {
  viennaps::gpu::impl::IonParams *params =
      (viennaps::gpu::impl::IonParams *)launchParams.customData;
  for (int i = 0; i < prd->ISCount; ++i) {
    auto geomNormal = computeNormal(sbtData, prd->TIndex[i]);
    auto cosTheta = -viennacore::DotProduct(prd->dir, geomNormal);
    float incomingAngle = acosf(max(min(cosTheta, 1.f), 0.f));

    float yield = 1.f;
    if (params->yieldFac >= 0.f) {
      yield = (params->yieldFac * cosTheta - 1.55f * cosTheta * cosTheta +
               0.65f * cosTheta * cosTheta * cosTheta) /
              (params->yieldFac - 0.9f);
    }

    atomicAdd(&launchParams.resultBuffer[getIdxOffset(0, launchParams) +
                                         prd->TIndex[i]],
              prd->rayWeight * yield);
  }
}

__forceinline__ __device__ void
faradayIonReflection(const void *sbtData, viennaray::gpu::PerRayData *prd) {
  viennaps::gpu::impl::IonParams *params =
      (viennaps::gpu::impl::IonParams *)launchParams.customData;
  auto geomNormal = computeNormal(sbtData, prd->primID);
  auto cosTheta = -viennacore::DotProduct(prd->dir, geomNormal);
  float incomingAngle = acosf(max(min(cosTheta, 1.f), 0.f));

  float sticking = 1.f;
  if (incomingAngle > params->thetaRMin)
    sticking = 1.f - min((incomingAngle - params->thetaRMin) /
                             (params->thetaRMax - params->thetaRMin),
                         1.f);
  prd->rayWeight -= prd->rayWeight * sticking;

  if (prd->rayWeight > launchParams.rayWeightThreshold) {
    conedCosineReflection(prd, geomNormal,
                          M_PI_2f - min(incomingAngle, params->minAngle),
                          launchParams.D);
  }
}
