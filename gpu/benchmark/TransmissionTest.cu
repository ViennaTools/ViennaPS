// #pragma once

#include <vcContext.hpp>
#include <vcVectorType.hpp>

#include <raygLaunchParams.hpp>
#include <raygReflection.hpp>

extern "C" __constant__ viennaray::gpu::LaunchParams launchParams;

// OptiX does not check for function signature, therefore
// the noop can take any parameters
extern "C" __device__ void __direct_callable__noop(void *, void *) {
  // does nothing
  // If a reflection is linked to this function, the program
  // will run indefinitely
}

extern "C" __device__ void
__direct_callable__transmissionTestCollision(const void *sbtData,
                                             viennaray::gpu::PerRayData *prd) {
  const int material = launchParams.materialIds[prd->primID];
  const uint3 idx = optixGetLaunchIndex();
  const uint3 dims = optixGetLaunchDimensions();
  const int linearLaunchIndex =
      idx.x + idx.y * dims.x + idx.z * dims.x * dims.y;
  const unsigned arrayIndex = linearLaunchIndex % launchParams.numElements;

  if (material == 1) {
    // Bottom (transmission)
    atomicAdd(&launchParams.resultBuffer[arrayIndex], 1.f);
    atomicAdd(
        &launchParams.resultBuffer[arrayIndex + 2 * launchParams.numElements],
        float(prd->numReflections));
  } else if (material == 2) {
    // Top (reflection)
    atomicAdd(&launchParams.resultBuffer[arrayIndex + launchParams.numElements],
              1.f);
    atomicAdd(
        &launchParams.resultBuffer[arrayIndex + 3 * launchParams.numElements],
        float(prd->numReflections));
  }
}

extern "C" __device__ void
__direct_callable__transmissionTestReflection(const void *sbtData,
                                              viennaray::gpu::PerRayData *prd) {
  int material = launchParams.materialIds[prd->primID];
  if (material == 1 || material == 2) {
    prd->rayWeight = 0.f;
  } else {
    auto geoNormal = viennaray::gpu::getNormal(sbtData, prd->primID);
    viennaray::gpu::diffuseReflection(prd, geoNormal, launchParams.D);
  }
}

extern "C" __device__ void
__direct_callable__transmissionTestInit(const void *sbtData,
                                        viennaray::gpu::PerRayData *prd) {
  const float radius = 10.f;
  auto distanceFromCenter =
      sqrtf(prd->pos[0] * prd->pos[0] + prd->pos[1] * prd->pos[1]);
  while (distanceFromCenter >= radius - 1e-3f) {
    // re-sample position
    const float4 u = curand_uniform4(&prd->RNGstate); // (0,1)
    prd->pos[0] = (u.x * 2.f - 1.f) * radius;
    prd->pos[1] = (u.y * 2.f - 1.f) * radius;
    distanceFromCenter =
        sqrtf(prd->pos[0] * prd->pos[0] + prd->pos[1] * prd->pos[1]);
  }
}
