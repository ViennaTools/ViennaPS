#include <optix_device.h>

#include <curtLaunchParams.hpp>
#include <curtPerRayData.hpp>
#include <curtRNG.hpp>
#include <curtReflection.hpp>
#include <curtSBTRecords.hpp>
#include <curtSource.hpp>
#include <curtUtilities.hpp>
#include <utGDT.hpp>

#include <context.hpp>

#define ION_PARTICLE_IDX 0
#define ETCHANT_PARTICLE_IDX 1
#define POLY_PARTICLE_IDX 2
#define ETCHANTPOLY_PARTICLE_IDX 3

extern "C" __constant__ curtLaunchParams<NumericType> params;
enum
{
  SURFACE_RAY_TYPE = 0,
  RAY_TYPE_COUNT
};

__constant__ NumericType sqrt_Eth_sp = 4.2426406871;
__constant__ NumericType sqrt_Eth_ie = 2.;
__constant__ NumericType sqrt_Eth_p = 2.;

__constant__ NumericType Ae_sp = 0.00339;
__constant__ NumericType Ae_ie = 0.0361;
__constant__ NumericType Ap_ie = 0.2888;
__constant__ NumericType B_sp = 9.3;

__constant__ NumericType Eref_max = 1.;

__constant__ NumericType minEnergy = 1.; // Discard particles with energy < 1eV

__constant__ NumericType inflectAngle = 1.55334;
__constant__ NumericType minAngle = 1.3962634;
__constant__ NumericType n_l = 10.;
__constant__ NumericType n_r = 1.;
__constant__ NumericType A_cont = 0.8989739349493948;

__constant__ NumericType gamma_p = 0.26;
__constant__ NumericType gamma_e = 0.9;
__constant__ NumericType gamma_pe = 0.6;

/* --------------- ION --------------- */

extern "C" __global__ void __closesthit__ion()
{
  const HitSBTData *sbtData = (const HitSBTData *)optixGetSbtDataPointer();
  PerRayData<NumericType> *prd =
      (PerRayData<NumericType> *)getPRD<PerRayData<NumericType>>();

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
    gdt::vec3f geomNormal =
        computeNormal(sbtData, optixGetPrimitiveIndex());
    auto cosTheta = -gdt::dot(prd->dir, geomNormal);
    cosTheta = max(min(cosTheta, 1.f), 0.f);

    NumericType angle = acosf(cosTheta);

    const NumericType sqrtE = sqrtf(prd->energy);
    const NumericType f_e_sp = (1 + B_sp * (1 - cosTheta * cosTheta)) * cosTheta;
    const NumericType Y_sp = Ae_sp * max(sqrtE - sqrt_Eth_sp, 0.f) * f_e_sp;
    const NumericType Y_ie = Ae_ie * max(sqrtE - sqrt_Eth_ie, 0.f) * cosTheta;
    const NumericType Y_p = Ap_ie * max(sqrtE - sqrt_Eth_p, 0.f) * cosTheta;

    // sputtering yield Y_sp ionSputteringFlux
    atomicAdd(&params.resultBuffer[getIdx(ION_PARTICLE_IDX, 0, &params)], prd->rayWeight * Y_sp);

    // ion enhanced etching yield Y_ie ionEnhancedFlux
    atomicAdd(&params.resultBuffer[getIdx(ION_PARTICLE_IDX, 1, &params)], prd->rayWeight * Y_ie);

    // ion enhanced O sputtering yield Y_O ionPolymerFlux
    atomicAdd(&params.resultBuffer[getIdx(ION_PARTICLE_IDX, 2, &params)], prd->rayWeight * Y_p);

    // ---------- REFLECTION ------------ //
    prd->rayWeight -= prd->rayWeight * cosTheta;
    NumericType Eref_peak = 0.f;

    if (angle >= inflectAngle)
    {
      Eref_peak =
          Eref_max *
          (1.f - (1.f - A_cont) *
                     pow((PI_F / 2.f - angle) / (PI_F / 2.f - inflectAngle), n_r));
    }
    else
    {
      Eref_peak = Eref_max * A_cont * pow(angle / inflectAngle, n_l);
    }
    // Gaussian distribution around the Eref_peak scaled by the particle energy
    NumericType tempEnergy = Eref_peak * prd->energy;
    NumericType NewEnergy;

    do
    {
      const auto rand1 = getNextRand(&prd->RNGstate);
      const auto rand2 = getNextRand(&prd->RNGstate);
      NewEnergy = tempEnergy + (min((prd->energy - tempEnergy), tempEnergy) + prd->energy * 0.05f) 
                             * cosf(2 * PI_F * rand1) * sqrtf(-2.f * logf(rand2));
    } while (NewEnergy > prd->energy || NewEnergy <= 0.f);

    // Set the flag to stop tracing if the energy is below the threshold
    if (NewEnergy > minEnergy)
    {
      prd->energy = NewEnergy;
      specularReflection(prd);
    }
    else
    {
      prd->energy = -1.f;
    }
  }
}

extern "C" __global__ void __miss__ion()
{
  getPRD<PerRayData<NumericType>>()->rayWeight = 0.f;
}

extern "C" __global__ void __raygen__ion()
{
  const uint3 idx = optixGetLaunchIndex();
  const uint3 dims = optixGetLaunchDimensions();
  const int linearLaunchIndex =
      idx.x + idx.y * dims.x + idx.z * dims.x * dims.y;

  // per-ray data
  PerRayData<NumericType> prd;
  prd.rayWeight = 1.f;
  // each ray has its own RNG state
  initializeRNGState(&prd, linearLaunchIndex, params.seed);

  // generate ray direction
  const NumericType sourcePower = params.cosineExponent;
  initializeRayRandom(&prd, &params, sourcePower, idx);

  do
  {
    const auto r = getNextRand(&prd.RNGstate);
    prd.energy = params.meanIonEnergy + (1.f / params.ionRF) * sinf(2.f * PI_F * r);
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

/* --------------- ETCHANT --------------- */

extern "C" __global__ void __closesthit__etchant()
{
  const HitSBTData *sbtData = (const HitSBTData *)optixGetSbtDataPointer();
  PerRayData<NumericType> *prd =
      (PerRayData<NumericType> *)getPRD<PerRayData<NumericType>>();

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
    NumericType *data = (NumericType *)sbtData->cellData;

    // const auto &phi_F = data[primID];
    // const auto &phi_O = data[primID + params.numElements];

    const NumericType Seff = gamma_e; // * max(1.f - phi_F - phi_O, 0.f);
    atomicAdd(&params.resultBuffer[getIdx(ETCHANT_PARTICLE_IDX, 0, &params)], prd->rayWeight * Seff);
    prd->rayWeight -= prd->rayWeight * Seff;
    diffuseReflection(prd);
  }
}

extern "C" __global__ void __miss__etchant()
{
  getPRD<PerRayData<NumericType>>()->rayWeight = 0.f;
}

extern "C" __global__ void __raygen__etchant()
{
  const uint3 idx = optixGetLaunchIndex();
  const uint3 dims = optixGetLaunchDimensions();
  const int linearLaunchIndex =
      idx.x + idx.y * dims.x + idx.z * dims.x * dims.y;

  // per-ray data
  PerRayData<NumericType> prd;
  prd.rayWeight = 1.f;
  // each ray has its own RNG state
  initializeRNGState(&prd, linearLaunchIndex, params.seed);

  // generate ray direction
  const NumericType sourcePower = 1.;
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

/* ------------- POLY --------------- */

extern "C" __global__ void __closesthit__polymer()
{
  const HitSBTData *sbtData = (const HitSBTData *)optixGetSbtDataPointer();
  PerRayData<NumericType> *prd =
      (PerRayData<NumericType> *)getPRD<PerRayData<NumericType>>();

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
    // NumericType *data = (NumericType *)sbtData->cellData;
    // const auto &phi_F = data[primID];
    // const auto &phi_O = data[primID + params.numElements];

    const NumericType Seff = gamma_p; // * max(1. - phi_F - phi_O, 0.);
    atomicAdd(&params.resultBuffer[getIdx(POLY_PARTICLE_IDX, 0, &params)], prd->rayWeight * Seff);
    prd->rayWeight -= prd->rayWeight * Seff;
    diffuseReflection(prd);
  }
}

extern "C" __global__ void __miss__polymer()
{
  getPRD<PerRayData<NumericType>>()->rayWeight = 0.f;
}

extern "C" __global__ void __raygen__polymer()
{
  const uint3 idx = optixGetLaunchIndex();
  const uint3 dims = optixGetLaunchDimensions();
  const int linearLaunchIndex =
      idx.x + idx.y * dims.x + idx.z * dims.x * dims.y;

  // per-ray data
  PerRayData<NumericType> prd;
  prd.rayWeight = 1.f;
  // each ray has its own RNG state
  initializeRNGState(&prd, linearLaunchIndex, params.seed);

  // generate ray direction
  const NumericType sourcePower = 1.;
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

/* ----------- ETCHANT POLY --------------- */

extern "C" __global__ void __closesthit__etchantPoly()
{
  const HitSBTData *sbtData = (const HitSBTData *)optixGetSbtDataPointer();
  PerRayData<NumericType> *prd =
      (PerRayData<NumericType> *)getPRD<PerRayData<NumericType>>();

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
    // NumericType *data = (NumericType *)sbtData->cellData;
    // const auto &phi_F = data[primID];
    // const auto &phi_O = data[primID + params.numElements];

    const NumericType Seff = gamma_pe; // * max(1. - phi_F - phi_O, 0.);
    atomicAdd(&params.resultBuffer[getIdx(ETCHANTPOLY_PARTICLE_IDX, 0, &params)], prd->rayWeight * Seff);
    prd->rayWeight -= prd->rayWeight * Seff;
    diffuseReflection(prd);
  }
}

extern "C" __global__ void __miss__etchantPoly()
{
  getPRD<PerRayData<NumericType>>()->rayWeight = 0.f;
}

extern "C" __global__ void __raygen__etchantPoly()
{
  const uint3 idx = optixGetLaunchIndex();
  const uint3 dims = optixGetLaunchDimensions();
  const int linearLaunchIndex =
      idx.x + idx.y * dims.x + idx.z * dims.x * dims.y;

  // per-ray data
  PerRayData<NumericType> prd;
  prd.rayWeight = 1.f;
  // each ray has its own RNG state
  initializeRNGState(&prd, linearLaunchIndex, params.seed);

  // generate ray direction
  const NumericType sourcePower = 1.;
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