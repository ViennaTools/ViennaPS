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

extern "C" __constant__ curtLaunchParams<float> params;
enum
{
  SURFACE_RAY_TYPE = 0,
  RAY_TYPE_COUNT
};

__constant__ float sqrt_Eth_sp = 4.2426406871;
__constant__ float sqrt_Eth_ie = 2.;
__constant__ float sqrt_Eth_p = 2.;

__constant__ float Ae_sp = 0.00339;
__constant__ float Ae_ie = 0.0361;
__constant__ float Ap_ie = 0.1444;
__constant__ float B_sp = 9.3;

__constant__ float Eref_max = 1.;

__constant__ float minEnergy = 1.; // Discard particles with energy < 1eV

__constant__ float inflectAngle = 1.55334;
__constant__ float minAngle = 1.3962634;
__constant__ float n_l = 10.;
__constant__ float n_r = 1.;
__constant__ float peak = 0.2;
__constant__ float A_cont = 0.8989739349493948;
__constant__ float minAvgConeAngle = 0.f;

__constant__ float gamma_p = 0.26;
__constant__ float gamma_e = 0.9;
__constant__ float gamma_pe = 0.6;

/* --------------- ION --------------- */

extern "C" __global__ void __closesthit__ion()
{
  const HitSBTData *sbtData = (const HitSBTData *)optixGetSbtDataPointer();
  PerRayData *prd =
      (PerRayData *)getPRD<PerRayData>();

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

    const float sqrtE = sqrtf(prd->energy);
    const float f_e_sp = (1 + B_sp * (1 - cosTheta * cosTheta)) * cosTheta;
    const float Y_sp = Ae_sp * max(sqrtE - sqrt_Eth_sp, 0.f) * f_e_sp;
    const float Y_ie = Ae_ie * max(sqrtE - sqrt_Eth_ie, 0.f) * cosTheta;
    const float Y_p = Ap_ie * max(sqrtE - sqrt_Eth_p, 0.f) * cosTheta;

    // sputtering yield Y_sp ionSputteringFlux
    atomicAdd(&params.resultBuffer[getIdx(ION_PARTICLE_IDX, 0, &params)], Y_sp);

    // ion enhanced etching yield Y_ie ionEnhancedFlux
    atomicAdd(&params.resultBuffer[getIdx(ION_PARTICLE_IDX, 1, &params)], Y_ie);

    // ion enhanced O sputtering yield Y_O ionPolymerFlux
    atomicAdd(&params.resultBuffer[getIdx(ION_PARTICLE_IDX, 2, &params)], Y_p);

    // ---------- REFLECTION ------------ //
    float Eref_peak = 0.f;
    float incAngle = acosf(cosTheta);

    if (incAngle >= inflectAngle)
    {
      Eref_peak =
          Eref_max *
          (1.f - (1.f - A_cont) *
                     pow((M_PI_2f - incAngle) / (M_PI_2f - inflectAngle), n_r));
    }
    else
    {
      Eref_peak = Eref_max * A_cont * pow(incAngle / inflectAngle, n_l);
    }
    // Gaussian distribution around the Eref_peak scaled by the particle energy
    float tempEnergy = Eref_peak * prd->energy;
    float NewEnergy;

    do
    {
      const auto rand1 = getNextRand(&prd->RNGstate);
      const auto rand2 = getNextRand(&prd->RNGstate);
      NewEnergy = tempEnergy + (min((prd->energy - tempEnergy), tempEnergy) +
                                prd->energy * 0.05f) *
                                   cosf(2 * M_PIf * rand1) * sqrtf(-2.f * logf(rand2));
    } while (NewEnergy > prd->energy || NewEnergy <= 0.f);

    // Set the flag to stop tracing if the energy is below the threshold
    if (NewEnergy > minEnergy)
    {
      prd->energy = NewEnergy;

      // coned cosine reflection
      specularReflection(prd, geomNormal);
      float avgReflAngle =
          max(M_PI_2f - incAngle, minAvgConeAngle);

      float u, sqrt_1m_u, angle;
      // generate a random angle between 0 and specular angle
      do
      {
        u = sqrtf(getNextRand(&prd->RNGstate));
        sqrt_1m_u = sqrtf(1. - u);
        angle = avgReflAngle * sqrt_1m_u;
      } while (getNextRand(&prd->RNGstate) * angle * u >
               cosf(M_PI_2f * sqrt_1m_u) * sinf(angle));

      cosTheta = cosf(angle);

      // Random Azimuthal Rotation
      float cosphi, sinphi;
      float r2;

      do
      {
        cosphi = getNextRand(&prd->RNGstate) - 0.5f;
        sinphi = getNextRand(&prd->RNGstate) - 0.5f;
        r2 = cosphi * cosphi + sinphi * sinphi;
      } while (r2 >= 0.25 || r2 <= 1e-4f);

      gdt::vec3f randomDir;

      // Rotate
      cosTheta = min(cosTheta, 1.);

      float a0;
      float a1;

      if (fabs(prd->dir[0]) <= fabs(prd->dir[1]))
      {
        a0 = prd->dir[0];
        a1 = prd->dir[1];
      }
      else
      {
        a0 = prd->dir[1];
        a1 = prd->dir[0];
      }

      const float a0_a0_m1 = 1. - a0 * a0;
      const float tmp =
          sqrtf(max(1. - cosTheta * cosTheta, 0.) / (r2 * a0_a0_m1));
      const float tmp_sinphi = tmp * sinphi;
      const float tmp_cosphi = tmp * cosphi;
      const float cosTheta_p_a0_tmp_sinphi = cosTheta + a0 * tmp_sinphi;

      randomDir[0] = a0 * cosTheta - a0_a0_m1 * tmp_sinphi;
      randomDir[1] = a1 * cosTheta_p_a0_tmp_sinphi + prd->dir[2] * tmp_cosphi;
      randomDir[2] = prd->dir[2] * cosTheta_p_a0_tmp_sinphi - a1 * tmp_cosphi;

      if (a0 != prd->dir[0])
      {
        // swap
        float tmp = randomDir[0];
        randomDir[0] = randomDir[1];
        randomDir[1] = tmp;
      }

      prd->dir = randomDir;
    }
    else
    {
      prd->energy = -1.f;
    }
  }
}

extern "C" __global__ void __miss__ion()
{
  getPRD<PerRayData>()->rayWeight = 0.f;
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

  // generate ray direction
  const float sourcePower = params.cosineExponent;
  initializeRayRandom(&prd, &params, sourcePower, idx);

  do
  {
    prd.energy = getNormalDistRand(&prd.RNGstate) * params.sigmaIonEnergy + params.meanIonEnergy;
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
  PerRayData *prd =
      (PerRayData *)getPRD<PerRayData>();

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
    float *data = (float *)sbtData->cellData;
    const float &phi_e = data[primID];

    atomicAdd(&params.resultBuffer[getIdx(ETCHANT_PARTICLE_IDX, 0, &params)], prd->rayWeight);

    const float Seff = gamma_e * max(1.f - phi_e, 0.f);
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
  prd.rayWeight = 1.f;
  // each ray has its own RNG state
  initializeRNGState(&prd, linearLaunchIndex, params.seed);

  // generate ray direction
  const float sourcePower = 1.;
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
  PerRayData *prd =
      (PerRayData *)getPRD<PerRayData>();

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
    atomicAdd(&params.resultBuffer[getIdx(POLY_PARTICLE_IDX, 0, &params)], prd->rayWeight);

    float *data = (float *)sbtData->cellData;
    const float &phi_e = data[primID];
    const float &phi_p = data[primID + params.numElements];
    const float &phi_pe = data[primID + 2 * params.numElements];

    const float Seff = gamma_pe * max(1.f - phi_e - phi_pe * phi_p, 0.f);
    prd->rayWeight -= prd->rayWeight * Seff;
    diffuseReflection(prd);
  }
}

extern "C" __global__ void __miss__polymer()
{
  getPRD<PerRayData>()->rayWeight = 0.f;
}

extern "C" __global__ void __raygen__polymer()
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

  // generate ray direction
  const float sourcePower = 1.;
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
  PerRayData *prd =
      (PerRayData *)getPRD<PerRayData>();

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

    float *data = (float *)sbtData->cellData;
    const auto &phi_pe = data[primID + 2 * params.numElements];

    atomicAdd(&params.resultBuffer[getIdx(ETCHANTPOLY_PARTICLE_IDX, 0, &params)], prd->rayWeight);
    const float Seff = gamma_pe * max(1.f - phi_pe, 0.f);
    prd->rayWeight -= prd->rayWeight * Seff;
    diffuseReflection(prd);
  }
}

extern "C" __global__ void __miss__etchantPoly()
{
  getPRD<PerRayData>()->rayWeight = 0.f;
}

extern "C" __global__ void __raygen__etchantPoly()
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

  // generate ray direction
  const float sourcePower = 1.;
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