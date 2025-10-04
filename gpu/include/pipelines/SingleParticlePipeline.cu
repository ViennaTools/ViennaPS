#include <optix_device.h>

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <raygBoundary.hpp>
#include <raygLaunchParams.hpp>
#include <raygPerRayData.hpp>
#include <raygRNG.hpp>
#include <raygReflection.hpp>
#include <raygSBTRecords.hpp>
#include <raygSource.hpp>

#include <vcContext.hpp>

// #ifndef COUNT_RAYS
// #define COUNT_RAYS
// #endif

using namespace viennaray::gpu;

/*  launch parameters in constant memory, filled in by optix upon
    optixLaunch (this gets filled in from the buffer we pass to
    optixLaunch) */
extern "C" __constant__ LaunchParams launchParams;

extern "C" __global__ void __closesthit__SingleParticle() {
  const HitSBTDataTriangle *sbtData = (const HitSBTDataTriangle *)optixGetSbtDataPointer();
  PerRayData *prd = (PerRayData *)getPRD<PerRayData>();

  if (sbtData->isBoundary) {
    if (launchParams.periodicBoundary) {
      applyPeriodicBoundary(prd, sbtData);
    } else {
      reflectFromBoundary(prd, sbtData);
    }
  } else {
    const unsigned int primID = optixGetPrimitiveIndex();
    atomicAdd(&launchParams.resultBuffer[primID], prd->rayWeight);
    prd->rayWeight -= prd->rayWeight * launchParams.sticking;
    if (prd->rayWeight > launchParams.rayWeightThreshold)
      diffuseReflection(prd);
  }
}

//------------------------------------------------------------------------------
// miss program that gets called for any ray that did not have a
// valid intersection
// ------------------------------------------------------------------------------
extern "C" __global__ void __miss__SingleParticle() {
  getPRD<PerRayData>()->rayWeight = 0.f;
}

//------------------------------------------------------------------------------
// ray gen program - entry point
//------------------------------------------------------------------------------
extern "C" __global__ void __raygen__SingleParticle() {
  const uint3 idx = optixGetLaunchIndex();
  const uint3 dims = optixGetLaunchDimensions();
  const int linearLaunchIndex =
      idx.x + idx.y * dims.x + idx.z * dims.x * dims.y;

  // per-ray data
  PerRayData prd;
  // each ray has its own RNG state
  initializeRNGState(&prd, linearLaunchIndex, launchParams.seed);

  // initialize ray position and direction
  initializeRayPosition(&prd, &launchParams);
  initializeRayDirection(&prd, launchParams.cosineExponent);

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
#ifdef COUNT_RAYS
    int *counter = reinterpret_cast<int *>(launchParams.customData);
    atomicAdd(counter, 1);
#endif
  }
}
