#pragma once

// #include "SingleParticle.cuh"
#include "MultiParticle.cuh"
// #include "PlasmaEtching.cuh"

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

// extern "C" __device__ void __direct_callable__singleNeutralCollision(
//     const viennaray::gpu::HitSBTData *sbtData,
//     viennaray::gpu::PerRayData *prd) {
//   singleNeutralCollision(prd);
// }

// extern "C" __device__ void __direct_callable__singleNeutralReflection(
//     const viennaray::gpu::HitSBTData *sbtData,
//     viennaray::gpu::PerRayData *prd) {
//   singleNeutralReflection(sbtData, prd);
// }

//
// --- MultParticle pipeline
//

extern "C" __device__ void __direct_callable__multiNeutralCollisionDisk(
    const viennaray::gpu::HitSBTDiskData *sbtData,
    viennaray::gpu::PerRayData *prd) {
  multiNeutralCollision(prd);
}

extern "C" __device__ void __direct_callable__multiNeutralCollisionTriangle(
    const viennaray::gpu::HitSBTData *sbtData,
    viennaray::gpu::PerRayData *prd) {
  multiNeutralCollision(prd);
}

extern "C" __device__ void __direct_callable__multiNeutralReflectionDisk(
    const viennaray::gpu::HitSBTDiskData *sbtData,
    viennaray::gpu::PerRayData *prd) {
  multiNeutralReflection(sbtData, prd);
}

extern "C" __device__ void __direct_callable__multiNeutralReflectionTriangle(
    const viennaray::gpu::HitSBTData *sbtData,
    viennaray::gpu::PerRayData *prd) {
  multiNeutralReflection(sbtData, prd);
}

extern "C" __device__ void __direct_callable__multiIonCollisionDisk(
    const viennaray::gpu::HitSBTDiskData *sbtData,
    viennaray::gpu::PerRayData *prd) {
  multiIonCollision(sbtData, prd);
}

extern "C" __device__ void __direct_callable__multiIonCollisionTriangle(
    const viennaray::gpu::HitSBTData *sbtData,
    viennaray::gpu::PerRayData *prd) {
  multiIonCollision(sbtData, prd);
}

extern "C" __device__ void __direct_callable__multiIonReflectionDisk(
    const viennaray::gpu::HitSBTDiskData *sbtData,
    viennaray::gpu::PerRayData *prd) {
  multiIonReflection(sbtData, prd);
}

extern "C" __device__ void __direct_callable__multiIonReflectionTriangle(
    const viennaray::gpu::HitSBTData *sbtData,
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

// extern "C" __device__ void __direct_callable__plasmaNeutralCollision(
//     const viennaray::gpu::HitSBTData *sbtData,
//     viennaray::gpu::PerRayData *prd) {
//   plasmaNeutralCollision(prd);
// }

// extern "C" __device__ void __direct_callable__plasmaNeutralReflection(
//     const viennaray::gpu::HitSBTData *sbtData,
//     viennaray::gpu::PerRayData *prd) {
//   plasmaNeutralReflection(sbtData, prd);
// }

// extern "C" __device__ void
// __direct_callable__plasmaIonCollision(const viennaray::gpu::HitSBTData
// *sbtData,
//                                       viennaray::gpu::PerRayData *prd) {
//   plasmaIonCollision(sbtData, prd);
// }

// extern "C" __device__ void __direct_callable__plasmaIonReflection(
//     const viennaray::gpu::HitSBTData *sbtData,
//     viennaray::gpu::PerRayData *prd) {
//   plasmaIonReflection(sbtData, prd);
// }

// extern "C" __device__ void
// __direct_callable__plasmaIonInit(const viennaray::gpu::HitSBTData *,
//                                  viennaray::gpu::PerRayData *prd) {
//   plasmaIonInit(prd);
// }