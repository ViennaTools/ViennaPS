#pragma once

#include <vcContext.hpp>
#include <vcVectorType.hpp>

#include <raygLaunchParams.hpp>
#include <raygReflection.hpp>

extern "C" __constant__ viennaray::gpu::LaunchParams launchParams;

//
// --- Ion particle
//

__forceinline__ __device__ void
TEOSPECVDIonReflection(const void *sbtData, viennaray::gpu::PerRayData *prd) {
  if (launchParams.sticking >= 1.f) {
    prd->rayWeight = 0.f; // terminate particle
    return;
  }

  prd->rayWeight -= prd->rayWeight * launchParams.sticking;

  float minAngle = *((float *)launchParams.customData);
  auto geoNormal = viennaray::gpu::getNormal(sbtData, prd->primID);
  auto cosTheta = __saturatef(
      -viennacore::DotProduct(prd->dir, geoNormal)); // clamp to [0,1]
  float theta = acosf(cosTheta);

  viennaray::gpu::conedCosineReflection(
      prd, geoNormal, M_PI_2f - min(theta, minAngle), launchParams.D);
}