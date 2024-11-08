#include <optix_device.h>

#include <curtLaunchParams.hpp>
#include <curtPerRayData.hpp>
#include <curtRNG.hpp>
#include <curtReflection.hpp>
#include <curtSBTRecords.hpp>
#include <curtSource.hpp>
#include <curtUtilities.hpp>
#include <utGDT.hpp>

#include <psConstants.hpp>

#include <context.hpp>

extern "C" __constant__ curtLaunchParams<float> params;
enum
{
  SURFACE_RAY_TYPE = 0,
  RAY_TYPE_COUNT
};

// __constant__ float A_sp = 0.00339;
__constant__ float A_Si = 7.;
// __constant__ float B_sp = 9.3;

// __constant__ float sqrt_Eth_p = 4.242640687;
__constant__ float sqrt_Eth_Si = 3.8729833462;
__constant__ float sqrt_Eth_O = 3.16227766;

__constant__ float Eref_max = 1.;

__constant__ float minEnergy = 4.; // Discard particles with energy < 4eV

__constant__ float inflectAngle = 1.55334;
__constant__ float minAngle = 1.3962634;
__constant__ float n_l = 10.;
__constant__ float n_r = 1.;
__constant__ float peak = 0.2;

__constant__ float gamma_O = 1.;
__constant__ float gamma_F = 0.7;

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
    gdt::vec3f geomNormal = computeNormal(sbtData, optixGetPrimitiveIndex());
    auto cosTheta = -gdt::dot(prd->dir, geomNormal);

    float angle = acosf(max(min(cosTheta, 1.f), 0.f));

    float f_ie_theta;
    if (cosTheta > 0.5f)
    {
      f_ie_theta = 1.f;
    }
    else
    {
      f_ie_theta = max(3.f - 6.f * angle / PI_F, 0.f);
    }
    float f_sp_theta =
        (1.f + psParameters::Si::B_sp * (1.f - cosTheta * cosTheta)) * cosTheta;

    float sqrtE = sqrtf(prd->energy);
    float Y_sp = psParameters::Si::A_sp *
                 max(sqrtE - psParameters::Si::Eth_sp_Ar_sqrt, 0.f) *
                 f_sp_theta;
    float Y_Si = A_Si * max(sqrtE - sqrt_Eth_Si, 0.f) * f_ie_theta;
    float Y_O = params.A_O * max(sqrtE - sqrt_Eth_O, 0.f) * f_ie_theta;

    // sputtering yield Y_sp ionSputteringRate
    atomicAdd(&params.resultBuffer[getIdx(0, 0, &params)],
              Y_sp * prd->rayWeight);

    // ion enhanced etching yield Y_Si ionEnhancedRate
    atomicAdd(&params.resultBuffer[getIdx(0, 1, &params)],
              Y_Si * prd->rayWeight);

    // ion enhanced O sputtering yield Y_O oxygenSputteringRate
    atomicAdd(&params.resultBuffer[getIdx(0, 2, &params)],
              Y_O * prd->rayWeight);

    // ------------- REFLECTION --------------- //

    // Small incident angles are reflected with the energy fraction centered at
    // 0
    float Eref_peak = 0.f;
    if (angle >= psParameters::Ion::inflectAngle)
    {
      Eref_peak = (1 - (1 - psParameters::Ion::A) * (M_PI_2f - angle) /
                           (M_PI_2f - psParameters::Ion::inflectAngle));
    }
    else
    {
      Eref_peak =
          psParameters::Ion::A *
          pow(angle / psParameters::Ion::inflectAngle, psParameters::Ion::n_l);
    }

    // Gaussian distribution around the Eref_peak scaled by the particle energy
    float NewEnergy;
    do
    {
      NewEnergy = getNormalDistRand(&prd->RNGstate) * prd->energy * 0.1f +
                  Eref_peak * prd->energy;
    } while (NewEnergy > prd->energy || NewEnergy <= 0.f);

    // Set the flag to stop tracing if the energy is below the threshold
    if (NewEnergy > minEnergy)
    {
      prd->energy = NewEnergy;
      prd->rayWeight -= prd->rayWeight * (1 - Eref_peak);
      specularReflection(prd, geomNormal);
    }
    else
    {
      prd->energy = -1.f;
    }
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
  // each ray has its own RNG state
  initializeRNGState(&prd, linearLaunchIndex, params.seed);

  // generate ray direction
  const float sourcePower = params.cosineExponent;
  initializeRayRandom(&prd, &params, sourcePower, idx);

  do
  {
    prd.energy = getNormalDistRand(&prd.RNGstate) * params.sigmaIonEnergy +
                 params.meanIonEnergy;
  } while (prd.energy < minEnergy);

  // the values we store the PRD pointer in:
  uint32_t u0, u1;
  packPointer((void *)&prd, u0, u1);

  while (prd.rayWeight > params.rayWeightThreshold && prd.energy > minEnergy)
  {
    optixTrace(params.traversable, // traversable GAS
               prd.pos,            // origin
               prd.dir,            // direction
               1e-4f,              // tmin
               1e20f,              // tmax
               0.0f,               // rayTime
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT, // OPTIX_RAY_FLAG_NONE,
               SURFACE_RAY_TYPE,              // SBT offset
               RAY_TYPE_COUNT,                // SBT stride
               SURFACE_RAY_TYPE,              // missSBTIndex
               u0, u1);
  }
}

extern "C" __global__ void __closesthit__etchant()
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
    float *data = (float *)sbtData->cellData;
    const unsigned int primID = optixGetPrimitiveIndex();
    const auto &phi_F = data[primID + params.numElements];
    const auto &phi_O = data[primID + 2 * params.numElements];

    const float Seff = gamma_F * max(1.f - phi_F - phi_O, 0.f);
    atomicAdd(&params.resultBuffer[getIdx(1, 0, &params)], prd->rayWeight);
    atomicAdd(&params.resultBuffer[getIdx(1, 1, &params)], Seff);
    prd->rayWeight -= prd->rayWeight * Seff;
    diffuseReflection(prd);
  }
}

extern "C" __global__ void __miss__etchant()
{
  getPRD<PerRayData>()->rayWeight = 0.f;
}

extern "C" __global__ void __raygen__etchant()
{
  const uint3 idx = optixGetLaunchIndex();
  const uint3 dims = optixGetLaunchDimensions();
  const int linearLaunchIndex =
      idx.x + idx.y * dims.x + idx.z * dims.x * dims.y;

  // per-ray data
  PerRayData prd;
  // each ray has its own RNG state
  initializeRNGState(&prd, linearLaunchIndex, params.seed);

  // generate ray direction
  const float sourcePower = 1.f;
  initializeRayRandom(&prd, &params, sourcePower, idx);

  // the values we store the PRD pointer in:
  uint32_t u0, u1;
  packPointer((void *)&prd, u0, u1);

  while (prd.rayWeight > params.rayWeightThreshold)
  {
    optixTrace(params.traversable, // traversable GAS
               prd.pos,            // origin
               prd.dir,            // direction
               1e-4f,              // tmin
               1e20f,              // tmax
               0.0f,               // rayTime
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT, // OPTIX_RAY_FLAG_NONE,
               SURFACE_RAY_TYPE,              // SBT offset
               RAY_TYPE_COUNT,                // SBT stride
               SURFACE_RAY_TYPE,              // missSBTIndex
               u0, u1);
  }
}

extern "C" __global__ void __closesthit__oxygen()
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
    float *data = (float *)sbtData->cellData;
    const unsigned int primID = optixGetPrimitiveIndex();
    const auto &phi_F = data[primID + params.numElements];
    const auto &phi_O = data[primID + 2 * params.numElements];

    const float Seff = gamma_O * max(1.f - phi_F - phi_O, 0.f);
    atomicAdd(&params.resultBuffer[getIdx(2, 0, &params)], prd->rayWeight);
    atomicAdd(&params.resultBuffer[getIdx(2, 1, &params)], Seff);
    prd->rayWeight -= prd->rayWeight * Seff;
    diffuseReflection(prd);
  }
}

extern "C" __global__ void __miss__oxygen()
{
  getPRD<PerRayData>()->rayWeight = 0.f;
}

extern "C" __global__ void __raygen__oxygen()
{
  const uint3 idx = optixGetLaunchIndex();
  const uint3 dims = optixGetLaunchDimensions();
  const int linearLaunchIndex =
      idx.x + idx.y * dims.x + idx.z * dims.x * dims.y;

  // per-ray data
  PerRayData prd;
  // each ray has its own RNG state
  initializeRNGState(&prd, linearLaunchIndex, params.seed);

  // generate ray direction
  const float sourcePower = 1.f;
  initializeRayRandom(&prd, &params, sourcePower, idx);

  // the values we store the PRD pointer in:
  uint32_t u0, u1;
  packPointer((void *)&prd, u0, u1);

  while (prd.rayWeight > params.rayWeightThreshold)
  {
    optixTrace(params.traversable, // traversable GAS
               prd.pos,            // origin
               prd.dir,            // direction
               1e-4f,              // tmin
               1e20f,              // tmax
               0.0f,               // rayTime
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT, // OPTIX_RAY_FLAG_NONE,
               SURFACE_RAY_TYPE,              // SBT offset
               RAY_TYPE_COUNT,                // SBT stride
               SURFACE_RAY_TYPE,              // missSBTIndex
               u0, u1);
  }
}
