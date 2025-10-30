#pragma once

#include <vcContext.hpp>
#include <vcVectorType.hpp>

#include "raygLaunchParams.hpp"
#include <raygReflection.hpp>

#include <psgPipelineParameters.hpp>

extern "C" __constant__ viennaray::gpu::LaunchParams launchParams;

//
// --- Ion particle
//

__forceinline__ __device__ void IBECollision(const void *sbtData,
                                             viennaray::gpu::PerRayData *prd) {
  viennaps::gpu::impl::IonParams *params =
      (viennaps::gpu::impl::IonParams *)launchParams.customData;
  const bool yieldDefined = abs(params->aSum) > 0.f;

  for (int i = 0; i < prd->ISCount; ++i) {
    auto geomNormal = computeNormal(sbtData, prd->primIDs[i]);
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

    // flux array
    atomicAdd(&launchParams.resultBuffer[getIdxOffset(0, launchParams) +
                                         prd->primIDs[i]],
              prd->rayWeight * yield);

    if (params->redepositionRate > 0.f) {
      // redeposition array
      atomicAdd(&launchParams.resultBuffer[getIdxOffset(1, launchParams) +
                                           prd->primIDs[i]],
                prd->load);
    }
  }
}

__forceinline__ __device__ void IBEReflection(const void *sbtData,
                                              viennaray::gpu::PerRayData *prd) {
  viennaps::gpu::impl::IonParams *params =
      (viennaps::gpu::impl::IonParams *)launchParams.customData;
  auto geomNormal = computeNormal(sbtData, prd->primID);
  auto cosTheta = __saturatef(
      -viennacore::DotProduct(prd->dir, geomNormal)); // clamp to [0,1]
  float theta = acosf(cosTheta);

  // Update redeposition weight
  if (params->redepositionRate > 0.f) {
    float yield = 1.f;
    if (abs(params->aSum) > 0.f) {
      float cosTheta2 = cosTheta * cosTheta;
      yield = (params->a1 * cosTheta + params->a2 * cosTheta2 +
               params->a3 * cosTheta2 * cosTheta +
               params->a4 * cosTheta2 * cosTheta2) /
              params->aSum;
    }
    yield *= max(sqrtf(prd->energy) - params->thresholdEnergy, 0.f);
    prd->load = yield;
  }

  float sticking = 1.f;
  if (theta > params->thetaRMin) {
    sticking = 1.f - __saturatef((theta - params->thetaRMin) /
                                 (params->thetaRMax - params->thetaRMin));
  }
  prd->rayWeight -= prd->rayWeight * sticking;

  if (prd->rayWeight < launchParams.rayWeightThreshold &&
      prd->load < params->redepositionThreshold) {
    return;
  }

  // Update energy
  viennaps::gpu::impl::updateEnergy(prd, params->inflectAngle, params->n_l,
                                    theta);

  if (prd->energy > params->thresholdEnergy * params->thresholdEnergy ||
      prd->load > params->redepositionThreshold) {
    conedCosineReflection(prd, geomNormal,
                          M_PI_2f - min(theta, params->minAngle),
                          launchParams.D);
  } else {
    prd->rayWeight = 0.f; // terminate particle
  }
}

__forceinline__ __device__ void IBEInit(viennaray::gpu::PerRayData *prd) {
  viennaps::gpu::impl::IonParams *params =
      (viennaps::gpu::impl::IonParams *)launchParams.customData;
  viennaps::gpu::impl::initNormalDistEnergy(
      prd, params->meanEnergy, params->sigmaEnergy,
      params->thresholdEnergy * params->thresholdEnergy);
  prd->load = 0.f;
}