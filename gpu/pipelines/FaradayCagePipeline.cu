#include <optix_device.h>

#include <curtLaunchParams.hpp>
#include <curtPerRayData.hpp>
#include <curtRNG.hpp>
#include <curtReflection.hpp>
#include <curtSBTRecords.hpp>
#include <curtSource.hpp>
#include <curtUtilities.hpp>

#include <context.hpp>

using namespace viennaps::gpu;
using namespace viennacore;

extern "C" __constant__ LaunchParams<float> params;
enum
{
  SURFACE_RAY_TYPE = 0,
  RAY_TYPE_COUNT
};

__constant__ float minAngle = 5 * M_PIf / 180.;
__constant__ float yieldFac = 1.075;
__constant__ float tiltAngle = 60.f;

extern "C" __global__ void __closesthit__ion()
{
  const HitSBTData *sbtData = (const HitSBTData *)optixGetSbtDataPointer();
  PerRayData *prd = (PerRayData *)getPRD<PerRayData>();

  if (sbtData->isBoundary)
  {
    if (params.periodicBoundary)
    {
      applyPeriodicBoundary(prd, sbtData);
    }
    else
    {
      reflectFromBoundary(prd);
    }
  }
  else
  {
    const unsigned int primID = optixGetPrimitiveIndex();
    Vec3Df geomNormal = computeNormal(sbtData, primID);
    float cosTheta = -DotProduct(prd->dir, geomNormal);
    cosTheta = max(min(cosTheta, 1.f), 0.f);
    // float incAngle = acosf(max(min(cosTheta, 1.f), 0.f));
    float yield = (yieldFac * cosTheta - 1.55 * cosTheta * cosTheta +
                   0.65 * cosTheta * cosTheta * cosTheta) /
                  (yieldFac - 0.9);

    atomicAdd(&params.resultBuffer[primID], prd->rayWeight * yield);

    // ---------- REFLECTION ------------ //
    prd->rayWeight -= prd->rayWeight * params.sticking;
    // conedCosineReflection(prd, (float)(M_PIf / 2.f - min(incAngle, minAngle)), geomNormal);
    specularReflection(prd);
  }
}

extern "C" __global__ void __miss__ion()
{
  getPRD<PerRayData>()->rayWeight = -1.f;
}

extern "C" __global__ void __raygen__ion()
{
  const uint3 idx = optixGetLaunchIndex();
  const uint3 dims = optixGetLaunchDimensions();
  const int linearLaunchIndex =
      idx.x + idx.y * dims.x + idx.z * dims.x * dims.y;

  // per-ray data
  PerRayData prd;
  prd.rayWeight = 1.f;
  // each ray has its own RNG state
  initializeRNGState(&prd, linearLaunchIndex, params.seed);

  // initialize ray position and direction
  initializeRayPosition(&prd, &params);
  initializeRayDirection(&prd, params.cosineExponent, params.source.directionBasis);

  // the values we store the PRD pointer in:
  uint32_t u0, u1;
  packPointer((void *)&prd, u0, u1);

  while (prd.rayWeight > params.rayWeightThreshold)
  {
    optixTrace(params.traversable,                              // traversable GAS
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