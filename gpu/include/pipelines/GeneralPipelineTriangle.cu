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

#include <models/psgPipelineParameters.hpp>

#include "CallableWrapper.cuh"

#include <vcContext.hpp>

using namespace viennaray::gpu;
using namespace viennacore;

extern "C" __constant__ viennaray::gpu::LaunchParams launchParams;

enum { SURFACE_RAY_TYPE = 0, RAY_TYPE_COUNT };

extern "C" __global__ void __closesthit__() {
  const HitSBTDataTriangle *sbtData = (const HitSBTDataTriangle *)optixGetSbtDataPointer();
  PerRayData *prd = (PerRayData *)getPRD<PerRayData>();

  const unsigned int primID = optixGetPrimitiveIndex();
  prd->tMin = optixGetRayTmax();
  prd->primID = primID;

  if (sbtData->base.isBoundary) {
    prd->numBoundaryHits++;
    // // This is effectively the miss shader
    // if (launchParams.D == 2 &&
    //     (primID == 2 || primID == 3)) { // bottom or top - ymin or ymax
    //   prd->rayWeight = 0.0f;
    //   return;
    // }
    // if (launchParams.D == 3 &&
    //     (primID == 4 || primID == 5)) { // bottom or top - zmin or zmax
    //   prd->rayWeight = 0.0f;
    //   return;
    // }
    if (launchParams.periodicBoundary) {
      applyPeriodicBoundary<viennaray::gpu::HitSBTDataTriangle>(prd, sbtData,
                                                        launchParams.D);
    } else {
      reflectFromBoundary<viennaray::gpu::HitSBTDataTriangle>(prd, sbtData,
                                                      launchParams.D);
    }
  } else {
    prd->ISCount = 1;
    prd->TIndex[0] = primID;

    // ------------- SURFACE COLLISION --------------- //
    unsigned callIdx;

    callIdx = callableIndex(launchParams.particleType, CallableSlot::COLLISION);
    optixDirectCall<void, const viennaray::gpu::HitSBTDataTriangle *, PerRayData *>(
        callIdx, sbtData, prd);

    // ------------- REFLECTION --------------- //

    callIdx =
        callableIndex(launchParams.particleType, CallableSlot::REFLECTION);
    optixDirectCall<void, const HitSBTDataTriangle *, PerRayData *>(callIdx, sbtData,
                                                            prd);
    prd->rayWeight = 0.f;
  }
}

extern "C" __global__ void __miss__() { getPRD<PerRayData>()->rayWeight = 0.f; }

extern "C" __global__ void __raygen__() {
  const uint3 idx = optixGetLaunchIndex();
  const uint3 dims = optixGetLaunchDimensions();
  const int linearLaunchIndex =
      idx.x + idx.y * dims.x + idx.z * dims.x * dims.y;

  // per-ray data
  PerRayData prd;
  // each ray has its own RNG state
  initializeRNGState(&prd, linearLaunchIndex, launchParams.seed);

  // initialize ray position and direction
  initializeRayPosition(&prd, &launchParams, launchParams.D);
  initializeRayDirection(&prd, launchParams.cosineExponent, launchParams.D);

  unsigned callIdx =
      callableIndex(launchParams.particleType, CallableSlot::INIT);
  optixDirectCall<void, const HitSBTDataDisk *, PerRayData *>(callIdx, nullptr,
                                                              &prd);

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
               SURFACE_RAY_TYPE,              // SBT offset
               RAY_TYPE_COUNT,                // SBT stride
               SURFACE_RAY_TYPE,              // missSBTIndex
               u0, u1);
  }
}
