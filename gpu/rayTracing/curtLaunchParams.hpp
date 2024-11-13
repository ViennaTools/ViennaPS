#pragma once

#include <optix_types.h>
#include <vcVectorUtil.hpp>

namespace viennaps {

namespace gpu {

template <typename T> struct LaunchParams {
  T *resultBuffer;
  T rayWeightThreshold = 0.01f;
  unsigned int seed = 0;
  unsigned int numElements;
  unsigned int *dataPerParticle;
  float sticking = 1.f;
  float cosineExponent = 1.f;
  bool periodicBoundary = true;
  float meanIonEnergy = 100.f; // eV
  float sigmaIonEnergy = 10.f; // eV
  float A_O = 3.f;

  // source plane params
  struct {
    viennacore::Vec2D<T> minPoint;
    viennacore::Vec2D<T> maxPoint;
    T gridDelta;
    T planeHeight;
  } source;

  OptixTraversableHandle traversable;
};

} // namespace gpu
} // namespace viennaps