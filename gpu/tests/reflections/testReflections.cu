#include <cuda.h>
#include <cuda_runtime.h>

#include <curtReflection.hpp>

#include <vcVectorUtil.hpp>

extern "C" __global__ void
test_diffuse(viennacore::Vec3Df inDir,
             viennacore::Vec3Df normal,
             viennacore::Vec3Df *results,
             unsigned numResults)
{
    using namespace viennacore;
    unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (; tidx < numResults; tidx += stride)
    {
        viennaps::gpu::PerRayData prd;
        initializeRNGState(&prd, tidx, 0);
        prd.dir = inDir;

        diffuseReflection(&prd, normal);
        results[tidx] = prd.dir;
    }
}

extern "C" __global__ void
test_coned_cosine(viennacore::Vec3Df inDir,
                  viennacore::Vec3Df normal,
                  float coneAngle,
                  viennacore::Vec3Df *results,
                  unsigned numResults)
{
    using namespace viennacore;
    unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (; tidx < numResults; tidx += stride)
    {
        viennaps::gpu::PerRayData prd;
        initializeRNGState(&prd, tidx, 0);
        prd.dir = inDir;

        conedCosineReflection(&prd, normal, coneAngle);
        results[tidx] = prd.dir;
    }
}