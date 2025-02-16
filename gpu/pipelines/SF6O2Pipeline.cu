#include <optix_device.h>

#include <curtLaunchParams.hpp>
#include <curtPerRayData.hpp>
#include <curtRNG.hpp>
#include <curtReflection.hpp>
#include <curtSBTRecords.hpp>
#include <curtSource.hpp>

#include <psSF6O2Parameters.hpp>

#include <gpu/vcContext.hpp>
#include <vcVectorUtil.hpp>

using namespace viennaps::gpu;

extern "C" __constant__ LaunchParams launchParams;

// for this simple example, we have a single ray type
enum
{
    SURFACE_RAY_TYPE = 0,
    RAY_TYPE_COUNT
};

extern "C" __global__ void __closesthit__ion()
{
    const HitSBTData *sbtData = (const HitSBTData *)optixGetSbtDataPointer();
    PerRayData *prd = (PerRayData *)getPRD<PerRayData>();

    if (sbtData->isBoundary)
    {
        if (launchParams.periodicBoundary)
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
        auto geomNormal = computeNormal(sbtData, optixGetPrimitiveIndex());
        auto cosTheta = -viennacore::DotProduct(prd->dir, geomNormal);
        viennaps::SF6O2Parameters<float> *params =
            reinterpret_cast<viennaps::SF6O2Parameters<float> *>(launchParams.customData);

        float angle = acosf(max(min(cosTheta, 1.f), 0.f));

        float f_ie_theta = 1.f;
        if (cosTheta <= 0.5f)
            f_ie_theta = max(3.f - 6.f * angle / M_PIf, 0.f);
        float f_sp_theta = 1.f;

        float sqrtE = sqrtf(prd->energy);
        float Y_sp = params->Si.A_sp * max(sqrtE - sqrtf(params->Si.Eth_sp), 0.f) * f_sp_theta;
        float Y_Si = params->Si.A_ie * max(sqrtE - sqrtf(params->Si.Eth_ie), 0.f) * f_ie_theta;
        float Y_O = params->Passivation.A_ie * max(sqrtE - sqrtf(params->Passivation.Eth_ie), 0.f) * f_ie_theta;

        // // sputtering yield Y_sp ionSputteringRate
        atomicAdd(&launchParams.resultBuffer[getIdx(0, 0, &launchParams)], Y_sp);

        // ion enhanced etching yield Y_Si ionEnhancedRate
        atomicAdd(&launchParams.resultBuffer[getIdx(0, 1, &launchParams)], Y_Si);

        // ion enhanced O sputtering yield Y_O oxygenSputteringRate
        atomicAdd(&launchParams.resultBuffer[getIdx(0, 2, &launchParams)], Y_O);

        // ------------- REFLECTION --------------- //

        // Small incident angles are reflected with the energy fraction centered at
        // 0
        float Eref_peak = 0.f;
        float A = 1. /
                  (1. + params->Ions.n_l * (M_PI_2f / params->Ions.inflectAngle - 1.));
        if (angle >= params->Ions.inflectAngle)
        {
            Eref_peak = 1 - (1 - A) * (M_PI_2f - angle) / (M_PI_2f - params->Ions.inflectAngle);
        }
        else
        {
            Eref_peak =
                A *
                pow(angle / params->Ions.inflectAngle, params->Ions.n_l);
        }

        // Gaussian distribution around the Eref_peak scaled by the particle energy
        float newEnergy;
        do
        {
            newEnergy = getNormalDistRand(&prd->RNGstate) * prd->energy * 0.1f +
                        Eref_peak * prd->energy;
        } while (newEnergy > prd->energy || newEnergy <= 0.f);

        // Set the flag to stop tracing if the energy is below the threshold
        float minEnergy = min(params->Si.Eth_ie, params->Si.Eth_sp);
        if (newEnergy > minEnergy)
        {
            prd->energy = newEnergy;
            conedCosineReflection(prd, geomNormal, M_PI_2f - min(angle, params->Ions.minAngle));
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
    // each ray has its own RNG state
    initializeRNGState(&prd, linearLaunchIndex, launchParams.seed);

    // initialize ray position and direction
    initializeRayPosition(&prd, &launchParams);
    initializeRayDirection(&prd, launchParams.cosineExponent);

    viennaps::SF6O2Parameters<float> *params =
        reinterpret_cast<viennaps::SF6O2Parameters<float> *>(launchParams.customData);
    float minEnergy = min(params->Si.Eth_ie, params->Si.Eth_sp);
    do
    {
        prd.energy = getNormalDistRand(&prd.RNGstate) * params->Ions.sigmaEnergy +
                     params->Ions.meanEnergy;
    } while (prd.energy < minEnergy);

    // the values we store the PRD pointer in:
    uint32_t u0, u1;
    packPointer((void *)&prd, u0, u1);

    while (prd.rayWeight > launchParams.rayWeightThreshold && prd.energy > minEnergy)
    {
        optixTrace(launchParams.traversable,                        // traversable GAS
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

extern "C" __global__ void __closesthit__etchant()
{
    const HitSBTData *sbtData = (const HitSBTData *)optixGetSbtDataPointer();
    PerRayData *prd = (PerRayData *)getPRD<PerRayData>();

    if (sbtData->isBoundary)
    {
        if (launchParams.periodicBoundary)
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
        const float &phi_F = data[primID];
        const float &phi_O = data[primID + launchParams.numElements];

        const float Seff = launchParams.sticking * max(1.f - phi_F - phi_O, 0.f);
        atomicAdd(&launchParams.resultBuffer[getIdx(1, 0, &launchParams)], prd->rayWeight);
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
    initializeRNGState(&prd, linearLaunchIndex, launchParams.seed);

    // initialize ray position and direction
    initializeRayPosition(&prd, &launchParams);
    initializeRayDirection(&prd, launchParams.cosineExponent);

    // the values we store the PRD pointer in:
    uint32_t u0, u1;
    packPointer((void *)&prd, u0, u1);

    while (prd.rayWeight > launchParams.rayWeightThreshold)
    {
        optixTrace(launchParams.traversable,                        // traversable GAS
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

extern "C" __global__ void __closesthit__oxygen()
{
    const HitSBTData *sbtData = (const HitSBTData *)optixGetSbtDataPointer();
    PerRayData *prd = (PerRayData *)getPRD<PerRayData>();

    if (sbtData->isBoundary)
    {
        if (launchParams.periodicBoundary)
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
        const auto &phi_F = data[primID];
        const auto &phi_O = data[primID + launchParams.numElements];

        const float Seff = launchParams.sticking * max(1.f - phi_F - phi_O, 0.f);
        atomicAdd(&launchParams.resultBuffer[getIdx(2, 0, &launchParams)], prd->rayWeight);
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
    initializeRNGState(&prd, linearLaunchIndex, launchParams.seed);

    // initialize ray position and direction
    initializeRayPosition(&prd, &launchParams);
    initializeRayDirection(&prd, launchParams.cosineExponent);

    // the values we store the PRD pointer in:
    uint32_t u0, u1;
    packPointer((void *)&prd, u0, u1);

    while (prd.rayWeight > launchParams.rayWeightThreshold)
    {
        optixTrace(launchParams.traversable,                        // traversable GAS
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
