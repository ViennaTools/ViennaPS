#include <optix_device.h>

#include <raygBoundary.hpp>
#include <raygLaunchParams.hpp>
#include <raygPerRayData.hpp>
#include <raygRNG.hpp>
#include <raygReflection.hpp>
#include <raygSBTRecords.hpp>
#include <raygSource.hpp>

#include <vcContext.hpp>

using namespace viennaray::gpu;
using namespace viennacore;

extern "C" __constant__ LaunchParams launchParams;

__constant__ float minAngle = 5 * M_PIf / 180.;
__constant__ float yieldFac = 1.075;
__constant__ float tiltAngle = 60.f;

extern "C" __global__ void __closesthit__ion() {
  const HitSBTData *sbtData = (const HitSBTData *)optixGetSbtDataPointer();
  PerRayData *prd = (PerRayData *)getPRD<PerRayData>();

  if (sbtData->isBoundary) {
    applyPeriodicBoundary(prd, sbtData);
    // if (launchParams.periodicBoundary)
    // {
    //   applyPeriodicBoundary(prd, sbtData);
    // }
    // else
    // {
    //   reflectFromBoundary(prd);
    // }
  } else {
    const unsigned int primID = optixGetPrimitiveIndex();
    Vec3Df geomNormal = computeNormal(sbtData, primID);
    float cosTheta = -DotProduct(prd->dir, geomNormal);
    cosTheta = max(min(cosTheta, 1.f), 0.f);
    // float incAngle = acosf(max(min(cosTheta, 1.f), 0.f));
    float yield = (yieldFac * cosTheta - 1.55 * cosTheta * cosTheta +
                   0.65 * cosTheta * cosTheta * cosTheta) /
                  (yieldFac - 0.9);

    atomicAdd(&launchParams.resultBuffer[primID], prd->rayWeight * yield);

    // ---------- REFLECTION ------------ //
    prd->rayWeight -= prd->rayWeight * launchParams.sticking;
    // conedCosineReflection(prd, (float)(M_PIf / 2.f - min(incAngle,
    // minAngle)), geomNormal);
    specularReflection(prd);
  }
}

extern "C" __global__ void __miss__ion() {
  getPRD<PerRayData>()->rayWeight = -1.f;
}

extern "C" __global__ void __raygen__ion() {
  const uint3 idx = optixGetLaunchIndex();
  const uint3 dims = optixGetLaunchDimensions();
  const int linearLaunchIndex =
      idx.x + idx.y * dims.x + idx.z * dims.x * dims.y;

  // per-ray data
  PerRayData prd;
  prd.rayWeight = 1.f;
  // each ray has its own RNG state
  initializeRNGState(&prd, linearLaunchIndex, launchParams.seed);

  // initialize ray position and direction
  initializeRayPosition(&prd, &launchParams);
  initializeRayDirection(&prd, launchParams.cosineExponent,
                         launchParams.source.directionBasis);

  // the values we store the PRD pointer in:
  uint32_t u0, u1;
  packPointer((void *)&prd, u0, u1);

  while (continueRay(launchParams, prd)) {
    optixTrace(launchParams.traversable, // traversable GAS
               make_float3(prd.pos[0], prd.pos[1], prd.pos[2]), // origin
               make_float3(prd.dir[0], prd.dir[1], prd.dir[2]), // direction
               1e-4f,                                           // tmin
               1e20f,                                           // tmax
               0.0f,                                            // rayTime
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT, // OPTIX_RAY_FLAG_NONE,
               0,                             // SBT offset
               1,                             // SBT stride
               0,                             // missSBTIndex
               u0, u1);
  }
}