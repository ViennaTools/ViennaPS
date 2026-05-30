#pragma once

#include "vcContext.hpp"
#include "vcVectorType.hpp"

#include "raygLaunchParams.hpp"
#include "raygReflection.hpp"

extern "C" __constant__ viennaray::gpu::LaunchParams launchParams;

namespace viennaps {

struct NeutralTransportParametersGPU {
  float etchFrontSticking = 1.f;
  float zeroCoverageSticking = 0.1f;
  int etchFrontMaterialId = 10; // Material::Si.legacyId()
  float desorptionRate = 0.f;
  float kEtch = 0.f;
  float surfaceSiteDensity = 1.66e-5f;
  float siliconDensity = 8.3e4f;
  float incomingFlux = 1.f;
};

} // namespace viennaps

//
// --- NeutralTransport particle
//

__forceinline__ __device__ void
neutralTransportCollision(viennaray::gpu::PerRayData *prd) {
  for (int i = 0; i < prd->ISCount; ++i) {
    atomicAdd(
        &launchParams
             .resultBuffer[viennaray::gpu::getIdxOffset(0, launchParams) +
                           prd->primIDs[i]],
        (viennaray::gpu::ResultType)prd->rayWeight);
  }
}

__forceinline__ __device__ void
neutralTransportReflection(const void *sbtData,
                           viennaray::gpu::PerRayData *prd) {
  const viennaps::NeutralTransportParametersGPU *params =
      reinterpret_cast<const viennaps::NeutralTransportParametersGPU *>(
          launchParams.customData);

  int consecutiveId = launchParams.materialIds[prd->primID];
  int legacyId = launchParams.materialMap[consecutiveId];

  if (legacyId == params->etchFrontMaterialId) {
    prd->rayWeight *= (1.f - __saturatef(params->etchFrontSticking));
    return; // skip normal/reflection computation for absorbed particles
  }

  const viennaray::gpu::HitSBTDataBase *baseData =
      reinterpret_cast<const viennaray::gpu::HitSBTDataBase *>(sbtData);
  const float *cellData = (const float *)baseData->cellData;
  const float theta = cellData[prd->primID];
  float sticking = params->zeroCoverageSticking * fmaxf(1.f - theta, 0.f);

  prd->rayWeight -= prd->rayWeight * __saturatef(sticking);
  auto geoNormal = viennaray::gpu::getNormal(sbtData, prd->primID);
  viennaray::gpu::diffuseReflection(prd, geoNormal);
}
