#include <optix_device.h>

#include <curtBoundary.hpp>
#include <curtLaunchParams.hpp>
#include <curtPerRayData.hpp>
#include <curtRNG.hpp>
#include <curtReflection.hpp>
#include <curtSBTRecords.hpp>
#include <curtSource.hpp>

#include <gpu/vcContext.hpp>

using namespace viennaps::gpu;

/*  launch parameters in constant memory, filled in by optix upon
    optixLaunch (this gets filled in from the buffer we pass to
    optixLaunch) */
extern "C" __constant__ LaunchParams<float> launchParams;

// for this simple example, we have a single ray type
enum
{
    SURFACE_RAY_TYPE = 0,
    RAY_TYPE_COUNT
};

extern "C" __global__ void __closesthit__SingleParticle()
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
        const unsigned int primID = optixGetPrimitiveIndex();
        atomicAdd(&launchParams.resultBuffer[primID], prd->rayWeight);
        prd->rayWeight -= prd->rayWeight * launchParams.sticking;
        if (prd->rayWeight > launchParams.rayWeightThreshold)
            diffuseReflection(prd);
    }
}

//------------------------------------------------------------------------------
// miss program that gets called for any ray that did not have a
// valid intersection
// ------------------------------------------------------------------------------
extern "C" __global__ void __miss__SingleParticle()
{
    getPRD<PerRayData>()->rayWeight = 0.f;
}

//------------------------------------------------------------------------------
// ray gen program - entry point
//------------------------------------------------------------------------------
extern "C" __global__ void __raygen__SingleParticle()
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
