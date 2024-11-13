#pragma once

#include <curtPerRayData.hpp>
#include <curtRNGState.hpp>

#ifdef __CUDACC__
template <unsigned int N>
static __device__ __inline__ unsigned int tea(unsigned int val0,
                                              unsigned int val1) {
  unsigned int v0 = val0;
  unsigned int v1 = val1;
  unsigned int s0 = 0;

  for (unsigned int n = 0; n < N; n++) {
    s0 += 0x9e3779b9;
    v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
    v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
  }

  return v0;
}

static __device__ void initializeRNGState(viennaps::gpu::PerRayData *prd,
                                          unsigned int linearLaunchIndex,
                                          unsigned int seed) {
  auto rngSeed = tea<4>(linearLaunchIndex, seed);
  curand_init(rngSeed, 0, 0, &prd->RNGstate);
}

__device__ float getNextRand(viennaps::gpu::RNGState *state) {
  return (float)(curand_uniform(state));
}

__device__ float getNormalDistRand(viennaps::gpu::RNGState *state) {
  float u0 = curand_uniform(state);
  float u1 = curand_uniform(state);
  float r = sqrtf(-2.f * logf(u0));
  float theta = 2.f * M_PIf * u1;
  return r * sinf(theta);
}

#endif