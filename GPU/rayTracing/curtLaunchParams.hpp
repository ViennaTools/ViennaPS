#pragma once

#include <optix_types.h>
#include <utGDT.hpp>

template <typename T> struct curtLaunchParams {
  T *resultBuffer;
  T rayWeightThreshold = 0.01f;
  unsigned int seed = 0;
  unsigned int numElements;
  unsigned int *dataPerParticle;
  float sticking = 1.f;
  float cosineExponent = 1.f;
  bool periodicBoundary = true;
  float meanIonEnergy = 100.f; // eV
  float ionRF = 0.1f;          // MHz
  float A_O = 3.f;

  // source plane params
  struct {
    gdt::vec_t<T, 2> minPoint;
    gdt::vec_t<T, 2> maxPoint;
    T gridDelta;
    T planeHeight;
  } source;

  OptixTraversableHandle traversable;
};
