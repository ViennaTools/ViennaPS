#pragma once

#include <optix_types.h>
#include <utGDT.hpp>

template <typename T> struct curtLaunchParams {
  T *resultBuffer;
  int voxelDim = 1;
  T rayWeightThreshold = 0.01f;
  unsigned int seed = 0;
  unsigned int numElements;
  unsigned int *dataPerParticle;
  float sticking = 1.f;
  float cosineExponent = 1.f;
  bool periodicBoundary = true;

  // source plane params
  struct {
    gdt::vec_t<T, 2> minPoint;
    gdt::vec_t<T, 2> maxPoint;
    T gridDelta;
    T planeHeight;
  } source;

  OptixTraversableHandle traversable;
};
