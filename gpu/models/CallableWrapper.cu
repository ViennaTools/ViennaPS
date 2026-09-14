// #pragma once

#include "FaradayCage.cuh"
#include "FluorocarbonEtching.cuh"
#include "IonBeamEtching.cuh"
#include "MultiParticle.cuh"
#include "NeutralTransport.cuh"
#include "PlasmaEtching.cuh"
#include "SingleParticle.cuh"
#include "SingleParticleALD.cuh"
#include "TEOSPECVD.cuh"

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
// --- NeutralTransport pipeline
//

extern "C" __device__ void __direct_callable__neutralTransportCollision(
    const void *sbtData, viennaray::gpu::PerRayData *prd, unsigned int primID) {
  neutralTransportCollision(prd, primID);
}

extern "C" __device__ void __direct_callable__neutralTransportReflection(
    const void *sbtData, viennaray::gpu::PerRayData *prd, unsigned int primID) {
  neutralTransportReflection(sbtData, prd, primID);
}

//
// --- SingleParticle pipeline
//

extern "C" __device__ void __direct_callable__singleNeutralCollision(
    const void *sbtData, viennaray::gpu::PerRayData *prd, unsigned int primID) {
  singleNeutralCollision(prd, primID);
}

extern "C" __device__ void __direct_callable__singleNeutralReflection(
    const void *sbtData, viennaray::gpu::PerRayData *prd, unsigned int primID) {
  singleNeutralReflection(sbtData, prd, primID);
}

extern "C" __device__ void __direct_callable__singleALDNeutralReflection(
    const void *sbtData, viennaray::gpu::PerRayData *prd, unsigned int primID) {
  singleALDNeutralReflection(sbtData, prd, primID);
}

extern "C" __device__ void __direct_callable__singleALDNeutralCollision(
    const void *sbtData, viennaray::gpu::PerRayData *prd, unsigned int primID) {
  singleALDNeutralCollision(sbtData, prd, primID);
}

//
// --- MultParticle pipeline
//

extern "C" __device__ void __direct_callable__multiNeutralCollision(
    const void *sbtData, viennaray::gpu::PerRayData *prd, unsigned int primID) {
  multiNeutralCollision(prd, primID);
}

extern "C" __device__ void __direct_callable__multiNeutralReflection(
    const void *sbtData, viennaray::gpu::PerRayData *prd, unsigned int primID) {
  multiNeutralReflection(sbtData, prd, primID);
}

extern "C" __device__ void __direct_callable__multiIonCollision(
    const void *sbtData, viennaray::gpu::PerRayData *prd, unsigned int primID) {
  multiIonCollision(sbtData, prd, primID);
}

extern "C" __device__ void __direct_callable__multiIonReflection(
    const void *sbtData, viennaray::gpu::PerRayData *prd, unsigned int primID) {
  multiIonReflection(sbtData, prd, primID);
}

extern "C" __device__ void
__direct_callable__multiIonInit(const void *, viennaray::gpu::PerRayData *prd) {
  multiIonInit(prd);
}

//
// --- PlasmaEtching pipeline
//

extern "C" __device__ void __direct_callable__plasmaNeutralCollision(
    const void *sbtData, viennaray::gpu::PerRayData *prd, unsigned int primID) {
  plasmaNeutralCollision(prd, primID);
}

extern "C" __device__ void __direct_callable__plasmaNeutralReflection(
    const void *sbtData, viennaray::gpu::PerRayData *prd, unsigned int primID) {
  plasmaNeutralReflection(sbtData, prd, primID);
}

extern "C" __device__ void
__direct_callable__plasmaNeutralReflectionNoPassivation(
    const void *sbtData, viennaray::gpu::PerRayData *prd, unsigned int primID) {
  plasmaNeutralReflectionNoPassivation(sbtData, prd, primID);
}

extern "C" __device__ void __direct_callable__plasmaIonCollision(
    const void *sbtData, viennaray::gpu::PerRayData *prd, unsigned int primID) {
  plasmaIonCollision(sbtData, prd, primID);
}

extern "C" __device__ void __direct_callable__plasmaIonReflection(
    const void *sbtData, viennaray::gpu::PerRayData *prd, unsigned int primID) {
  plasmaIonReflection(sbtData, prd, primID);
}

extern "C" __device__ void
__direct_callable__plasmaIonInit(const void *,
                                 viennaray::gpu::PerRayData *prd) {
  plasmaIonInit(prd);
}

//
// --- IonBeamEtching pipeline
//

extern "C" __device__ void __direct_callable__IBECollision(
    const void *sbtData, viennaray::gpu::PerRayData *prd, unsigned int primID) {
  IBECollision(sbtData, prd, primID);
}

extern "C" __device__ void __direct_callable__IBEReflection(
    const void *sbtData, viennaray::gpu::PerRayData *prd, unsigned int primID) {
  IBEReflection(sbtData, prd, primID);
}

extern "C" __device__ void
__direct_callable__IBEInit(const void *, viennaray::gpu::PerRayData *prd) {
  IBEInit(prd);
}

//
// --- FaradayCage pipeline
//

extern "C" __device__ void __direct_callable__faradayCollision(
    const void *sbtData, viennaray::gpu::PerRayData *prd, unsigned int primID) {
  faradayIonCollision(sbtData, prd, primID);
}

//
// --- TEOSPECVD pipeline
//

extern "C" __device__ void __direct_callable__TEOSPECVDIonReflection(
    const void *sbtData, viennaray::gpu::PerRayData *prd, unsigned int primID) {
  TEOSPECVDIonReflection(sbtData, prd, primID);
}

//
// --- FluorocarbonEtching pipeline
//

extern "C" __device__ void __direct_callable__fluorocarbonNeutralCollision(
    const void *sbtData, viennaray::gpu::PerRayData *prd, unsigned int primID) {
  fluorocarbonNeutralCollision(prd, primID);
}

extern "C" __device__ void __direct_callable__fluorocarbonNeutralReflection(
    const void *sbtData, viennaray::gpu::PerRayData *prd, unsigned int primID) {
  fluorocarbonNeutralReflection(sbtData, prd, primID);
}

extern "C" __device__ void __direct_callable__fluorocarbonIonCollision(
    const void *sbtData, viennaray::gpu::PerRayData *prd, unsigned int primID) {
  fluorocarbonIonCollision(sbtData, prd, primID);
}

extern "C" __device__ void __direct_callable__fluorocarbonIonReflection(
    const void *sbtData, viennaray::gpu::PerRayData *prd, unsigned int primID) {
  fluorocarbonIonReflection(sbtData, prd, primID);
}

extern "C" __device__ void
__direct_callable__fluorocarbonIonInit(const void *,
                                       viennaray::gpu::PerRayData *prd) {
  fluorocarbonIonInit(prd);
}