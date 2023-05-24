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

extern "C" __constant__ curtLaunchParams<NumericType> params;
enum
{
  SURFACE_RAY_TYPE = 0,
  RAY_TYPE_COUNT
};

__constant__ NumericType A_sp = 0.00339;
__constant__ NumericType A_Si = 7.;
__constant__ NumericType B_sp = 9.3;

__constant__ NumericType sqrt_Eth_p = 4.242640687;
__constant__ NumericType sqrt_Eth_Si = 3.8729833462;
__constant__ NumericType sqrt_Eth_O = 3.16227766;

__constant__ NumericType Eref_max = 1.;

__constant__ NumericType minEnergy = 1.; // Discard particles with energy < 4eV

__constant__ NumericType inflectAngle = 1.55334;
__constant__ NumericType minAngle = 1.3962634;
__constant__ NumericType n_l = 10.;
__constant__ NumericType n_r = 1.;
__constant__ NumericType peak = 0.2;

__constant__ NumericType gamma_O = 1.;
__constant__ NumericType gamma_F = 0.1;

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

    NumericType angle = acosf(max(min(cosTheta, 1.f), 0.f));

    NumericType f_sp_theta;
    NumericType f_Si_theta;
    NumericType f_O_theta;

    if (cosTheta > 0.5f)
    {
      f_Si_theta = 1.f;
      f_O_theta = 1.f;
    }
    else
    {
      f_Si_theta = max(3.f - 6.f * angle / PI_F, 0.f);
      f_O_theta = max(3.f - 6.f * angle / PI_F, 0.f);
    }
    f_sp_theta = (1 + B_sp * (1 - cosTheta * cosTheta)) * cosTheta;

    NumericType sqrtE = sqrt(prd->energy);
    NumericType Y_sp = A_sp * max(sqrtE - sqrt_Eth_p, 0.f) * f_sp_theta;
    NumericType Y_Si = A_Si * max(sqrtE - sqrt_Eth_Si, 0.f) * f_Si_theta;
    NumericType Y_O = params.A_O * max(sqrtE - sqrt_Eth_O, 0.f) * f_O_theta;

    // sputtering yield Y_sp ionSputteringRate
    atomicAdd(&params.resultBuffer[getIdx(0, 0, &params)], Y_sp);

    // ion enhanced etching yield Y_Si ionEnhancedRate
    atomicAdd(&params.resultBuffer[getIdx(0, 1, &params)], Y_Si);

    // ion enhanced O sputtering yield Y_O oxygenSputteringRate
    atomicAdd(&params.resultBuffer[getIdx(0, 2, &params)], Y_O);

    // ---------- REFLECTION ------------ //
    NumericType Eref_peak = 0.f;

    // Small incident angles are reflected with the energy fraction centered at
    // 0
    const NumericType A =
        1.f / (1.f + (n_l / n_r) * (PI_F / (2.f * inflectAngle) - 1.f));
    if (angle >= inflectAngle)
    {
      Eref_peak =
          Eref_max *
          (1.f - (1.f - A) *
                     pow((PI_F / 2.f - angle) / (PI_F / 2.f - inflectAngle), n_r));
    }
    else
    {
      Eref_peak = Eref_max * A * pow(angle / inflectAngle, n_l);
    }
    // Gaussian distribution around the Eref_peak scaled by the particle energy
    NumericType tempEnergy = Eref_peak * prd->energy;
    NumericType NewEnergy;

    do
    {
      const auto rand1 = getNextRand(&prd->RNGstate);
      const auto rand2 = getNextRand(&prd->RNGstate);
      NewEnergy = tempEnergy + (min((prd->energy - tempEnergy), tempEnergy) +
                                prd->energy * 0.05f) *
                                   cosf(2 * PI_F * rand1) * sqrtf(-2.f * logf(rand2));
    } while (NewEnergy > prd->energy || NewEnergy <= 0.f);

    // Set the flag to stop tracing if the energy is below the threshold
    if (NewEnergy > minEnergy)
    {
      prd->energy = NewEnergy;
      // conedCosineReflection(prd, (NumericType)(PI_F / 2.f - min(incAngle,
      // minAngle)), geomNormal);
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
  getPRD<PerRayData<NumericType>>()->rayWeight = -1.f;
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
    NumericType rand1 = r * (2.f * PI_F - 2 * peak) + peak;
    prd.energy = (1 + cosf(rand1)) * (params.ionRF / 2 * 0.75 + 10);
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
    NumericType *data = (NumericType *)sbtData->cellData;
    const unsigned int primID = optixGetPrimitiveIndex();
    const auto &phi_F = data[primID];
    const auto &phi_O = data[primID + params.numElements];

    const NumericType Seff = gamma_F; //  * max(1.f - phi_F - phi_O, 0.f);
    atomicAdd(&params.resultBuffer[getIdx(1, 0, &params)], prd->rayWeight);
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

extern "C" __global__ void __closesthit__oxygen()
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
    NumericType *data = (NumericType *)sbtData->cellData;
    const unsigned int primID = optixGetPrimitiveIndex();
    const auto &phi_F = data[primID];
    const auto &phi_O = data[primID + params.numElements];

    const NumericType Seff = gamma_O;
    // *max(1. - phi_F - phi_O, 0.);
    atomicAdd(&params.resultBuffer[getIdx(2, 0, &params)], prd->rayWeight);
    prd->rayWeight -= prd->rayWeight * Seff;
    diffuseReflection(prd);
  }
}

extern "C" __global__ void __miss__oxygen()
{
  getPRD<PerRayData<NumericType>>()->rayWeight = 0.f;
}

extern "C" __global__ void __raygen__oxygen()
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
