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

#include <vcContext.hpp>

using namespace viennaray::gpu;

extern "C" __constant__ LaunchParams launchParams;

enum { SURFACE_RAY_TYPE = 0, RAY_TYPE_COUNT };

extern "C" __global__ void __closesthit__Neutral() {
  const HitSBTData *sbtData = (const HitSBTData *)optixGetSbtDataPointer();
  PerRayData *prd = (PerRayData *)getPRD<PerRayData>();

  if (sbtData->isBoundary) {
    if (launchParams.periodicBoundary) {
      applyPeriodicBoundary(prd, sbtData);
    } else {
      reflectFromBoundary(prd);
    }
  } else {
    const unsigned int primID = optixGetPrimitiveIndex();
    atomicAdd(&launchParams.resultBuffer[primID], prd->rayWeight);
    prd->rayWeight -= prd->rayWeight * launchParams.sticking;
    if (prd->rayWeight > launchParams.rayWeightThreshold)
      diffuseReflection(prd);
  }
}

extern "C" __global__ void __miss__Neutral() {
  getPRD<PerRayData>()->rayWeight = 0.f;
}

extern "C" __global__ void __raygen__Neutral() {
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

extern "C" __global__ void __closesthit__Ion() {
  const HitSBTData *sbtData = (const HitSBTData *)optixGetSbtDataPointer();
  PerRayData *prd = (PerRayData *)getPRD<PerRayData>();

  if (sbtData->isBoundary) {
    if (launchParams.periodicBoundary) {
      applyPeriodicBoundary(prd, sbtData);
    } else {
      reflectFromBoundary(prd);
    }
  } else {
    viennaps::gpu::impl::IonParams *params =
        (viennaps::gpu::impl::IonParams *)launchParams.customData;
    auto geomNormal = computeNormal(sbtData, optixGetPrimitiveIndex());
    auto cosTheta = -viennacore::DotProduct(prd->dir, geomNormal);
    float incomingAngle = acosf(max(min(cosTheta, 1.f), 0.f));

    // ------------- SURFACE COLLISION ------------- //
    float flux = prd->rayWeight;
    if (params->B_sp >= 0.f) {
      flux *= (1 + params->B_sp * (1 - cosTheta * cosTheta)) * cosTheta;
    }

    if (params->meanEnergy > 0.f) {
      flux *= max(sqrtf(prd->energy) - sqrtf(params->thresholdEnergy), 0.f);
    }

    atomicAdd(&launchParams.resultBuffer[getIdx(0, launchParams)], flux);

    // ------------- REFLECTION ------------- //

    float sticking = 1.;
    if (incomingAngle > params->thetaRMin)
      sticking = 1.f - min((incomingAngle - params->thetaRMin) /
                               (params->thetaRMax - params->thetaRMin),
                           1.f);
    prd->rayWeight -= prd->rayWeight * sticking;

    if (params->meanEnergy > 0.f) {
      float Eref_peak;
      float A = 1. / (1. + params->n * (M_PI_2 / params->inflectAngle - 1.));
      if (incomingAngle >= params->inflectAngle) {
        Eref_peak = (1 - (1 - A) * (M_PI_2 - incomingAngle) /
                             (M_PI_2 - params->inflectAngle));
      } else {
        Eref_peak = A * powf(incomingAngle / params->inflectAngle, params->n);
      }

      float newEnergy;
      do {
        newEnergy = getNormalDistRand(&prd->RNGstate) * prd->energy * 0.1f +
                    Eref_peak * prd->energy;
      } while (newEnergy > prd->energy || newEnergy <= 0.f);

      prd->energy = newEnergy;
    }

    if (prd->rayWeight > launchParams.rayWeightThreshold &&
        prd->energy > params->thresholdEnergy) {
      conedCosineReflection(prd, geomNormal,
                            M_PI_2f - min(incomingAngle, params->minAngle));
    }
  }
}

extern "C" __global__ void __miss__Ion() {
  getPRD<PerRayData>()->rayWeight = 0.f;
}

extern "C" __global__ void __raygen__Ion() {
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

  viennaps::gpu::impl::IonParams *params =
      (viennaps::gpu::impl::IonParams *)launchParams.customData;

  if (params->meanEnergy > 0.f) {
    do {
      prd.energy = getNormalDistRand(&prd.RNGstate) * params->sigmaEnergy +
                   params->meanEnergy;
    } while (prd.energy <= 0.);
  } else {
    prd.energy = std::numeric_limits<float>::max();
  }

  // the values we store the PRD pointer in:
  uint32_t u0, u1;
  packPointer((void *)&prd, u0, u1);

  while (prd.rayWeight > launchParams.rayWeightThreshold &&
         prd.energy > params->thresholdEnergy) {
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
