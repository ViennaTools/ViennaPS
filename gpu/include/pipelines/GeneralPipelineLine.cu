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

extern "C" __global__ void __intersection__() {
  const HitSBTDataLine *sbtData =
      (const HitSBTDataLine *)optixGetSbtDataPointer();
  PerRayData *prd = (PerRayData *)getPRD<PerRayData>();

  // Get the index of the AABB box that was hit
  const int primID = optixGetPrimitiveIndex();

  // Read geometric data from the primitive that is inside that AABB box
  const Vec2D<unsigned> idx = (sbtData->lines)[primID];
  const Vec3Df p0 = sbtData->nodes[idx[0]];
  const Vec3Df p1 = sbtData->nodes[idx[1]];

  Vec3Df lineDir = p1 - p0;
  Vec3Df d = CrossProduct(prd->dir, lineDir);

  bool valid = true;

  float t = CrossProduct((p0 - prd->pos), lineDir)[2] / d[2];
  valid &= t > 1e-4f;
  float s = CrossProduct((p0 - prd->pos), prd->dir)[2] / d[2];
  valid &= s > 1e-4f && s < 1.0f - 1e-4f;

  if (valid) {
    // Collect all intersections and filter out overlapping lines in CH shader
    if (!sbtData->base.isBoundary && prd->tempCount < MAX_NEIGHBORS) {
      prd->tValues[prd->tempCount] = t;
      prd->primIDs[prd->tempCount] = primID;
      prd->tempCount++;
    }

    // Has to pass a dummy t value so later intersections are not ignored
    optixReportIntersection(t + prd->tThreshold, 0);
  }
}

extern "C" __global__ void __closesthit__() {
  const HitSBTDataLine *sbtData =
      (const HitSBTDataLine *)optixGetSbtDataPointer();
  PerRayData *prd = (PerRayData *)getPRD<PerRayData>();

  const unsigned int primID = optixGetPrimitiveIndex();
  prd->tMin = optixGetRayTmax() - prd->tThreshold;
  prd->primID = primID;

  const Vec3Df normal = computeNormal(sbtData, primID);

  // If closest hit was on backside, let it through once
  if (DotProduct(prd->dir, normal) > 0.0f) {
    // If back was hit a second time, kill the ray
    if (prd->hitFromBack) {
      prd->rayWeight = 0.f;
      return;
    }
    prd->hitFromBack = true;
    prd->pos = prd->pos + prd->tMin * prd->dir;
    return;
  }

  if (sbtData->base.isBoundary) {
    prd->numBoundaryHits++;
    // This is effectively the miss shader
    if (primID == 2 || primID == 3) { // bottom or top - ymin or ymax
      prd->rayWeight = 0.0f;
      return;
    }

    if (launchParams.periodicBoundary) {
      applyPeriodicBoundary(prd, sbtData, launchParams.D);
    } else {
      reflectFromBoundary(prd, sbtData, launchParams.D);
    }
  } else {
    // ------------- NEIGHBOR FILTERING --------------- //
    // Keep only hits close to tMin
    prd->ISCount = 0;
    for (int i = 0; i < prd->tempCount; ++i) {
      if (fabsf(prd->tValues[i] - prd->tMin) < prd->tThreshold &&
          prd->ISCount < MAX_NEIGHBORS) {
        prd->TIndex[prd->ISCount++] = prd->primIDs[i];
      }
    }

    // // CPU like neighbor detection
    // prd->ISCount = 0;
    // for (int i = 0; i < prd->tempCount; ++i) {
    //   float distance = viennacore::Distance(sbtData->point[primID],
    //                                         sbtData->point[prd->primIDs[i]]);
    //   if (distance < 2 * sbtData->radius && prd->ISCount < MAX_NEIGHBORS) {
    //     prd->TIndex[prd->ISCount++] = prd->primIDs[i];
    //   }
    // }

    // // Actual equivalent to CPU version
    // prd->TIndex[0] = primID;
    // prd->ISCount = 1;
    // for (int i = 0; i < launchParams.maxNeighbors; ++i) {
    //   int neighborIdx =
    //       launchParams.neighbors[primID * launchParams.maxNeighbors + i];
    //   if (neighborIdx == -1)
    //     continue;
    //   const Vec3Df diskOrigin = sbtData->point[neighborIdx];
    //   const Vec3Df normal = sbtData->normal[neighborIdx];
    //   const float radius = sbtData->radius;

    //   bool valid = true;
    //   float prodOfDirections = DotProduct(normal, prd->dir);
    //   // valid &= DotProduct(prd->dir, normal) <= 0.0f;
    //   valid &= fabsf(prodOfDirections) >= 1e-6f;

    //   float ddneg = DotProduct(diskOrigin, normal);
    //   float t = (ddneg - DotProduct(normal, prd->pos)) / prodOfDirections;
    //   valid &= t > 1e-4f;

    //   const Vec3Df intersection = prd->pos + prd->dir * t;
    //   const Vec3Df diff = intersection - diskOrigin;
    //   float distance = DotProduct(diff, diff);
    //   valid &= distance < radius * radius;
    //   if (valid)
    //     prd->TIndex[prd->ISCount++] = neighborIdx;
    // }

    // ------------- SURFACE COLLISION --------------- //
    unsigned callIdx;

    callIdx = callableIndex(launchParams.particleType, CallableSlot::COLLISION);
    optixDirectCall<void, const HitSBTDataLine *, PerRayData *>(callIdx,
                                                                sbtData, prd);

    // ------------- REFLECTION --------------- //
    callIdx =
        callableIndex(launchParams.particleType, CallableSlot::REFLECTION);
    optixDirectCall<void, const HitSBTDataLine *, PerRayData *>(callIdx,
                                                                sbtData, prd);
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
  prd.tThreshold = 1.1f * launchParams.gridDelta;
  // prd.tThreshold = 0.f;
  // each ray has its own RNG state
  initializeRNGState(&prd, linearLaunchIndex, launchParams.seed);

  // initialize ray position and direction
  initializeRayPosition(&prd, &launchParams, launchParams.D);
  initializeRayDirection(&prd, launchParams.cosineExponent, launchParams.D);

  unsigned callIdx =
      callableIndex(launchParams.particleType, CallableSlot::INIT);
  optixDirectCall<void, const HitSBTDataLine *, PerRayData *>(callIdx, nullptr,
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
    prd.tempCount = 0; // Reset PerRayData
  }
}
