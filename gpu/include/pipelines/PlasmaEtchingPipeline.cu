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

#include <models/psPlasmaEtchingParameters.hpp>

#include <vcContext.hpp>
#include <vcVectorType.hpp>

using namespace viennaray::gpu;

extern "C" __constant__ LaunchParams launchParams;

// for this simple example, we have a single ray type
enum { SURFACE_RAY_TYPE = 0, RAY_TYPE_COUNT };

extern "C" __global__ void __closesthit__ion() {
  const HitSBTDataTriangle *sbtData = (const HitSBTDataTriangle *)optixGetSbtDataPointer();
  PerRayData *prd = (PerRayData *)getPRD<PerRayData>();

  if (sbtData->isBoundary) {
    if (launchParams.periodicBoundary) {
      applyPeriodicBoundary(prd, sbtData);
    } else {
      reflectFromBoundary(prd, sbtData);
    }
  } else {
    auto primID = optixGetPrimitiveIndex();
    int material = launchParams.materialIds[primID];
    viennaps::PlasmaEtchingParameters<float> *params =
        reinterpret_cast<viennaps::PlasmaEtchingParameters<float> *>(
            launchParams.customData);

    auto geomNormal = computeNormal(sbtData, primID);
    auto cosTheta = -viennacore::DotProduct(prd->dir, geomNormal);
    float angle = acosf(max(min(cosTheta, 1.f), 0.f));

    float A_sp = params->Substrate.A_sp;
    float B_sp = params->Substrate.B_sp;
    float Eth_sp = params->Substrate.Eth_sp;
    if (material == 0) {
      // mask
      A_sp = params->Mask.A_sp;
      B_sp = params->Mask.B_sp;
      Eth_sp = params->Mask.Eth_sp;
    }

    float f_ie_theta = 1.f;
    if (cosTheta <= 0.5f)
      f_ie_theta = max(3.f - 6.f * angle / M_PIf, 0.f);
    float f_sp_theta =
        max((1.f + B_sp * (1 - cosTheta * cosTheta)) * cosTheta, 0.f);

    float sqrtE = sqrtf(prd->energy);
    float Y_sp = A_sp * max(sqrtE - sqrtf(Eth_sp), 0.f) * f_sp_theta;
    float Y_Si = params->Substrate.A_ie *
                 max(sqrtE - sqrtf(params->Substrate.Eth_ie), 0.f) * f_ie_theta;
    float Y_P = params->Passivation.A_ie *
                max(sqrtE - sqrtf(params->Passivation.Eth_ie), 0.f) *
                f_ie_theta;

    // // sputtering yield Y_sp ionSputteringRate
    atomicAdd(&launchParams.resultBuffer[getIdx(0, launchParams)], Y_sp);

    // ion enhanced etching yield Y_Si ionEnhancedRate
    atomicAdd(&launchParams.resultBuffer[getIdx(1, launchParams)], Y_Si);

    // ion enhanced O sputtering yield Y_P oxygenSputteringRate
    atomicAdd(&launchParams.resultBuffer[getIdx(2, launchParams)], Y_P);

    // ------------- REFLECTION --------------- //

    // Small incident angles are reflected with the energy fraction centered at
    // 0
    float Eref_peak = 0.f;
    float A = 1.f / (1.f + params->Ions.n_l *
                               (M_PI_2f / params->Ions.inflectAngle - 1.f));
    if (angle >= params->Ions.inflectAngle) {
      Eref_peak = 1.f - (1.f - A) * (M_PI_2f - angle) /
                            (M_PI_2f - params->Ions.inflectAngle);
    } else {
      Eref_peak = A * pow(angle / params->Ions.inflectAngle, params->Ions.n_l);
    }

    // Gaussian distribution around the Eref_peak scaled by the particle energy
    float newEnergy;
    do {
      newEnergy = getNormalDistRand(&prd->RNGstate) * prd->energy * 0.1f +
                  Eref_peak * prd->energy;
    } while (newEnergy > prd->energy || newEnergy <= 0.f);

    float sticking = 1.f;
    if (angle > params->Ions.thetaRMin) {
      sticking =
          1.f - max(min((angle - params->Ions.thetaRMin) /
                            (params->Ions.thetaRMax - params->Ions.thetaRMin),
                        1.f),
                    0.f);
    }
    prd->rayWeight -= prd->rayWeight * sticking;

    // Set the flag to stop tracing if the energy is below the threshold
    float minEnergy = min(params->Substrate.Eth_ie, params->Substrate.Eth_sp);
    if (newEnergy > minEnergy) {
      prd->energy = newEnergy;
      conedCosineReflection(prd, geomNormal,
                            M_PI_2f - min(angle, params->Ions.minAngle));
    } else {
      prd->energy = -1.f;
    }
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

  viennaps::PlasmaEtchingParameters<float> *params =
      reinterpret_cast<viennaps::PlasmaEtchingParameters<float> *>(
          launchParams.customData);
  float minEnergy = min(params->Substrate.Eth_ie, params->Substrate.Eth_sp);
  do {
    prd.energy = getNormalDistRand(&prd.RNGstate) * params->Ions.sigmaEnergy +
                 params->Ions.meanEnergy;
  } while (prd.energy < minEnergy);

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

extern "C" __global__ void __closesthit__neutral() {
  const HitSBTDataTriangle *sbtData = (const HitSBTDataTriangle *)optixGetSbtDataPointer();
  PerRayData *prd = (PerRayData *)getPRD<PerRayData>();

  if (sbtData->isBoundary) {
    if (launchParams.periodicBoundary) {
      applyPeriodicBoundary(prd, sbtData);
    } else {
      reflectFromBoundary(prd, sbtData);
    }
  } else {
    atomicAdd(&launchParams.resultBuffer[getIdx(0, launchParams)],
              prd->rayWeight);

    // ------------- REFLECTION --------------- //
    float *data = (float *)sbtData->cellData;
    const unsigned int primID = optixGetPrimitiveIndex();
    const auto &phi_E = data[primID];
    const auto &phi_P = data[primID + launchParams.numElements];
    int material = launchParams.materialIds[primID];
    float sticking = launchParams.materialSticking[material];
    const float Seff = sticking * max(1.f - phi_E - phi_P, 0.f);
    prd->rayWeight -= prd->rayWeight * Seff;
    diffuseReflection(prd);
  }
}

extern "C" __global__ void __miss__neutral() {
  getPRD<PerRayData>()->rayWeight = 0.f;
}

extern "C" __global__ void __raygen__neutral() {
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
               SURFACE_RAY_TYPE,              // SBT offset
               RAY_TYPE_COUNT,                // SBT stride
               SURFACE_RAY_TYPE,              // missSBTIndex
               u0, u1);
  }
}
