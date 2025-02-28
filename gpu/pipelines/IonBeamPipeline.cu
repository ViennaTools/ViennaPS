#include <optix_device.h>

#include <gpu/raygLaunchParams.hpp>
#include <gpu/raygPerRayData.hpp>
#include <gpu/raygRNG.hpp>
#include <gpu/raygReflection.hpp>
#include <gpu/raygSBTRecords.hpp>
#include <gpu/raygSource.hpp>

#include <psSF6O2Parameters.hpp>

#include <gpu/vcContext.hpp>
#include <vcVectorUtil.hpp>

using namespace viennaray::gpu;

extern "C" __constant__ LaunchParams launchParams;
__constant__ float minAngle = 75 * M_PIf / 180.;
__constant__ float thetaRMin = 60 * M_PIf / 180.;
__constant__ float thetaRMax = M_PI_2f;

// for this simple example, we have a single ray type
enum { SURFACE_RAY_TYPE = 0, RAY_TYPE_COUNT };

extern "C" __global__ void __closesthit__ion() {
  const HitSBTData *sbtData = (const HitSBTData *)optixGetSbtDataPointer();
  PerRayData *prd = (PerRayData *)getPRD<PerRayData>();

  if (sbtData->isBoundary) {
    if (launchParams.periodicBoundary) {
      applyPeriodicBoundary(prd, sbtData);
    } else {
      reflectFromBoundary(prd);
    }
  } else {
    auto geomNormal = computeNormal(sbtData, optixGetPrimitiveIndex());
    auto cosTheta = -viennacore::DotProduct(prd->dir, geomNormal);
    float angle = acosf(max(min(cosTheta, 1.f), 0.f));

    float f_ie_theta = 1.f;
    // if (cosTheta <= 0.5f)
    //     f_ie_theta = max(3.f - 6.f * angle / M_PIf, 0.f);

    float yield = prd->rayWeight * f_ie_theta;

    const unsigned int primID = optixGetPrimitiveIndex();
    atomicAdd(&launchParams.resultBuffer[primID], yield);

    // ------------- REFLECTION --------------- //

    float sticking = 1.;
    if (angle > thetaRMin)
      sticking = 1.f - min((angle - thetaRMin) / (thetaRMax - thetaRMin), 1.f);

    prd->rayWeight -= prd->rayWeight * sticking;
    if (prd->rayWeight > launchParams.rayWeightThreshold)
      conedCosineReflection(prd, geomNormal, M_PI_2f - min(angle, minAngle));
  }
}

extern "C" __global__ void __miss__ion() {
  getPRD<PerRayData>()->rayWeight = 0.f;
}

extern "C" __global__ void __raygen__ion() {
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

  while (prd.rayWeight > launchParams.rayWeightThreshold) {
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
