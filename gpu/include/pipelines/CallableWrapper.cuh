#pragma once

#include "SingleParticle.cuh"
#include "MultiParticle.cuh"
#include "PlasmaEtching.cuh"

//
// --- Direct Callables wrapper
//
// - Direct callables must not call any OptiX API functions
//   (e.g. OptixGetPrimitiveIndex(), etc.))
// - Every wrapper must have the exact same signature

extern "C" __device__ void
__direct_callable__noop(const viennaray::gpu::HitSBTData *,
                        viennaray::gpu::PerRayData *) {
  // does nothing
  // If a reflection is linked to this function, the program
  // will run indefinitely
}

//
// --- SingleParticle pipeline
//

extern "C" __device__ void __direct_callable__singleNeutralCollision(
    const viennaray::gpu::HitSBTData *sbtData,
    viennaray::gpu::PerRayData *prd) {
  singleNeutralCollision(prd);
}

extern "C" __device__ void __direct_callable__singleNeutralReflection(
    const viennaray::gpu::HitSBTData *sbtData,
    viennaray::gpu::PerRayData *prd) {
  singleNeutralReflection(sbtData, prd);
}

//
// --- MultParticle pipeline
//

extern "C" __device__ void __direct_callable__multiNeutralCollision(
    const viennaray::gpu::HitSBTData *sbtData,
    viennaray::gpu::PerRayData *prd) {
  multiNeutralCollision(prd);
}

extern "C" __device__ void __direct_callable__multiNeutralReflection(
    const viennaray::gpu::HitSBTData *sbtData,
    viennaray::gpu::PerRayData *prd) {
  multiNeutralReflection(sbtData, prd);
}

extern "C" __device__ void
__direct_callable__multiIonCollision(const viennaray::gpu::HitSBTData *sbtData,
                                     viennaray::gpu::PerRayData *prd) {
  multiIonCollision(sbtData, prd);
}

extern "C" __device__ void
__direct_callable__multiIonReflection(const viennaray::gpu::HitSBTData *sbtData,
                                      viennaray::gpu::PerRayData *prd) {
  multiIonReflection(sbtData, prd);
}

extern "C" __device__ void
__direct_callable__multiIonInit(const viennaray::gpu::HitSBTData *,
                                viennaray::gpu::PerRayData *prd) {
  multiIonInit(prd);
}

//
// --- PlasmaEtching pipeline
//

extern "C" __device__ void __direct_callable__plasmaNeutralCollision(
    const viennaray::gpu::HitSBTData *sbtData,
    viennaray::gpu::PerRayData *prd) {
  plasmaNeutralCollision(prd);
}

extern "C" __device__ void __direct_callable__plasmaNeutralReflection(
    const viennaray::gpu::HitSBTData *sbtData,
    viennaray::gpu::PerRayData *prd) {
  plasmaNeutralReflection(sbtData, prd);
}

extern "C" __device__ void
__direct_callable__plasmaIonCollision(const viennaray::gpu::HitSBTData *sbtData,
                                      viennaray::gpu::PerRayData *prd) {
  plasmaIonCollision(sbtData, prd);
}

extern "C" __device__ void __direct_callable__plasmaIonReflection(
    const viennaray::gpu::HitSBTData *sbtData,
    viennaray::gpu::PerRayData *prd) {
  plasmaIonReflection(sbtData, prd);
}

extern "C" __device__ void
__direct_callable__plasmaIonInit(const viennaray::gpu::HitSBTData *,
                                 viennaray::gpu::PerRayData *prd) {
  plasmaIonInit(prd);
}