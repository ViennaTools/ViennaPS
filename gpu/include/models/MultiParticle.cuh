#pragma once

#include <vcContext.hpp>
#include <vcVectorType.hpp>

#include <raygLaunchParams.hpp>
#include <raygReflection.hpp>

#include <psgPipelineParameters.hpp>

extern "C" __constant__ viennaray::gpu::LaunchParams launchParams;

//
// --- Neutral particle
//

__forceinline__ __device__ void
multiNeutralCollision(viennaray::gpu::PerRayData *prd) {
  for (int i = 0; i < prd->ISCount; ++i) {
    atomicAdd(&launchParams
                   .resultBuffer[viennaray::gpu::getIdxOffset(0, launchParams) +
                                 prd->primIDs[i]],
              prd->rayWeight);
  }
}

__forceinline__ __device__ void
multiNeutralReflection(const void *sbtData, viennaray::gpu::PerRayData *prd) {
  int material = launchParams.materialIds[prd->primID];
  float sticking = launchParams.materialSticking[material];
  prd->rayWeight -= prd->rayWeight * sticking;
  auto geoNormal = viennaray::gpu::getNormal(sbtData, prd->primID);
  viennaray::gpu::diffuseReflection(prd, geoNormal, launchParams.D);
}

//
// --- Ion particle
//

__forceinline__ __device__ void
multiIonCollision(const void *sbtData, viennaray::gpu::PerRayData *prd) {
  viennaps::gpu::impl::IonParams *params =
      (viennaps::gpu::impl::IonParams *)launchParams.customData;
  for (int i = 0; i < prd->ISCount; ++i) {

    float flux = prd->rayWeight;

    if (params->B_sp >= 0.f) {
      auto geomNormal = viennaray::gpu::getNormal(sbtData, prd->primIDs[i]);
      auto cosTheta = __saturatef(
          -viennacore::DotProduct(prd->dir, geomNormal)); // clamp to [0,1]
      flux *= (1 + params->B_sp * (1.f - cosTheta * cosTheta)) * cosTheta;
    }

    if (params->meanEnergy > 0.f) {
      flux *= max(sqrtf(prd->energy) - params->thresholdEnergy, 0.f);
    }

    atomicAdd(&launchParams
                   .resultBuffer[viennaray::gpu::getIdxOffset(0, launchParams) +
                                 prd->primIDs[i]],
              flux);
  }
}

__forceinline__ __device__ void
multiIonReflection(const void *sbtData, viennaray::gpu::PerRayData *prd) {
  viennaps::gpu::impl::IonParams *params =
      (viennaps::gpu::impl::IonParams *)launchParams.customData;
  auto geomNormal = viennaray::gpu::getNormal(sbtData, prd->primID);
  auto cosTheta = __saturatef(-viennacore::DotProduct(prd->dir, geomNormal));
  float incomingAngle = acosf(cosTheta);

  float sticking = 1.f;
  if (incomingAngle > params->thetaRMin) {
    sticking = 1.f - __saturatef((incomingAngle - params->thetaRMin) /
                                 (params->thetaRMax - params->thetaRMin));
  }

  if (sticking >= 1.f) {
    prd->rayWeight = 0.f;
    return;
  }

  if (params->meanEnergy > 0.f) {
    viennaps::gpu::impl::updateEnergy(prd, params->inflectAngle, params->n_l,
                                      incomingAngle);
  }

  prd->rayWeight -= prd->rayWeight * sticking;
  conedCosineReflection(prd, geomNormal,
                        M_PI_2f - min(incomingAngle, params->minAngle),
                        launchParams.D);
}

__forceinline__ __device__ void multiIonInit(viennaray::gpu::PerRayData *prd) {
  viennaps::gpu::impl::IonParams *params =
      (viennaps::gpu::impl::IonParams *)launchParams.customData;
  if (params->meanEnergy > 0.f) {
    viennaps::gpu::impl::initNormalDistEnergy(prd, params->meanEnergy,
                                              params->sigmaEnergy);
  }
}