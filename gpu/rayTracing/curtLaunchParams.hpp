#pragma once

#include <optix_types.h>
#include <vcVectorUtil.hpp>

namespace viennaps {

namespace gpu {

struct LaunchParams {
  float *resultBuffer;
  float rayWeightThreshold = 0.01f;
  unsigned int seed = 0;
  unsigned int numElements;
  unsigned int *dataPerParticle; // to determine result buffer index
  float sticking = 1.f;
  float cosineExponent = 1.f;
  bool periodicBoundary = true;
  unsigned int maxRayDepth = 1000;
  void *customData;

  // source plane params
  struct {
    viennacore::Vec2Df minPoint;
    viennacore::Vec2Df maxPoint;
    float gridDelta;
    float planeHeight;
    std::array<viennacore::Vec3Df, 3> directionBasis;
  } source;

  OptixTraversableHandle traversable;
};

#ifdef __CUDACC__
__device__ __forceinline__ unsigned int getIdx(int particleIdx, int dataIdx,
                                               LaunchParams *launchParams) {
  unsigned int offset = 0;
  for (unsigned int i = 0; i < particleIdx; i++)
    offset += launchParams->dataPerParticle[i];
  offset = (offset + dataIdx) * launchParams->numElements;
  return offset + optixGetPrimitiveIndex();
}
#endif

} // namespace gpu
} // namespace viennaps