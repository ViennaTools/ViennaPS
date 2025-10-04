#pragma once

#include "FaradayCage.cuh"
#include "MultiParticle.cuh"
#include "PlasmaEtching.cuh"
#include "SingleParticle.cuh"
#include "SingleParticleALD.cuh"

//
// --- Direct Callables wrapper
//
// - Direct callables must not call any OptiX API functions
//   (e.g. OptixGetPrimitiveIndex(), etc.)
// - Every wrapper must take the same amount of parameters

// OptiX does not check for function signature, therefore
// the noop can take any parameters
extern "C" __device__ void __direct_callable__noop(void *, void *) {
  // does nothing
  // If a reflection is linked to this function, the program
  // will run indefinitely
}

//
// --- SingleParticle pipeline
//

extern "C" __device__ void
__direct_callable__singleNeutralCollision(const void *sbtData,
                                          viennaray::gpu::PerRayData *prd) {
  singleNeutralCollision(prd);
}

extern "C" __device__ void
__direct_callable__singleNeutralReflection(const void *sbtData,
                                           viennaray::gpu::PerRayData *prd) {
  singleNeutralReflection(sbtData, prd);
}

extern "C" __device__ void
__direct_callable__singleALDNeutralCollision(const void *sbtData,
                                             viennaray::gpu::PerRayData *prd) {
  singleALDNeutralCollision(sbtData, prd);
}

//
// --- MultParticle pipeline
//

extern "C" __device__ void
__direct_callable__multiNeutralCollision(const void *sbtData,
                                         viennaray::gpu::PerRayData *prd) {
  multiNeutralCollision(prd);
}

extern "C" __device__ void
__direct_callable__multiNeutralReflection(const void *sbtData,
                                          viennaray::gpu::PerRayData *prd) {
  multiNeutralReflection(sbtData, prd);
}

extern "C" __device__ void
__direct_callable__multiIonCollision(const void *sbtData,
                                     viennaray::gpu::PerRayData *prd) {
  multiIonCollision(sbtData, prd);
}

extern "C" __device__ void
__direct_callable__multiIonReflection(const void *sbtData,
                                      viennaray::gpu::PerRayData *prd) {
  multiIonReflection(sbtData, prd);
}

extern "C" __device__ void
__direct_callable__multiIonInit(const void *, viennaray::gpu::PerRayData *prd) {
  multiIonInit(prd);
}

//
// --- PlasmaEtching pipeline
//

extern "C" __device__ void
__direct_callable__plasmaNeutralCollision(const void *sbtData,
                                          viennaray::gpu::PerRayData *prd) {
  plasmaNeutralCollision(prd);
}

extern "C" __device__ void
__direct_callable__plasmaNeutralReflection(const void *sbtData,
                                           viennaray::gpu::PerRayData *prd) {
  plasmaNeutralReflection(sbtData, prd);
}

extern "C" __device__ void
__direct_callable__plasmaIonCollision(const void *sbtData,
                                      viennaray::gpu::PerRayData *prd) {
  plasmaIonCollision(sbtData, prd);
}

extern "C" __device__ void
__direct_callable__plasmaIonReflection(const void *sbtData,
                                       viennaray::gpu::PerRayData *prd) {
  plasmaIonReflection(sbtData, prd);
}

extern "C" __device__ void
__direct_callable__plasmaIonInit(const void *,
                                 viennaray::gpu::PerRayData *prd) {
  plasmaIonInit(prd);
}

//
// --- FaradayCage pipeline
//

extern "C" __device__ void
__direct_callable__faradayCollision(const void *sbtData,
                                    viennaray::gpu::PerRayData *prd) {
  faradayIonCollision(sbtData, prd);
}

extern "C" __device__ void
__direct_callable__faradayReflection(const void *sbtData,
                                     viennaray::gpu::PerRayData *prd) {
  faradayIonReflection(sbtData, prd);
}