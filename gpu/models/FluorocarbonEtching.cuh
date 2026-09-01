#pragma once

#include "vcContext.hpp"
#include "vcVectorType.hpp"

#include "raygLaunchParams.hpp"
#include "raygReflection.hpp"

#include "materials/psMaterialMap.hpp"
#include "models/psFluorocarbonParameters.hpp"
#include "models/psIonModelUtil.hpp"

extern "C" __constant__ viennaray::gpu::LaunchParams launchParams;

//
// --- Neutral particle
//

__forceinline__ __device__ void
fluorocarbonNeutralCollision(viennaray::gpu::PerRayData *prd) {
  for (int i = 0; i < prd->ISCount; ++i) {
    atomicAdd(&launchParams
                   .resultBuffer[viennaray::gpu::getIdxOffset(0, launchParams) +
                                 prd->primIDs[i]],
              (viennaray::gpu::ResultType)prd->rayWeight);
  }
}

__forceinline__ __device__ void
fluorocarbonNeutralReflection(const void *sbtData,
                              viennaray::gpu::PerRayData *prd) {
  const viennaray::gpu::HitSBTDataBase *baseData =
      reinterpret_cast<const viennaray::gpu::HitSBTDataBase *>(sbtData);

  float *coverages = (float *)baseData->cellData;
  float phi_E = coverages[prd->primID];
  float phi_P = coverages[prd->primID + launchParams.numElements];

  int id = launchParams.materialIds[prd->primID]; // consecutive ID, not enum
  float sticking = launchParams.materialSticking[id];

  float Seff = sticking * max(1.f - phi_E - phi_P, 0.f);
  prd->rayWeight -= prd->rayWeight * Seff;

  auto geoNormal = viennaray::gpu::getNormal(sbtData, prd->primID);
  viennaray::gpu::diffuseReflection(prd, geoNormal);
}

//
// --- Ion particle
//

__forceinline__ __device__ void
fluorocarbonIonCollision(const void *sbtData, viennaray::gpu::PerRayData *prd) {
  using namespace viennaps;
  gpu::FluorocarbonParameters *params =
      reinterpret_cast<gpu::FluorocarbonParameters *>(launchParams.customData);
  for (int i = 0; i < prd->ISCount; ++i) {
    int id = launchParams.materialIds[prd->primIDs[i]]; // consecutive ID
    int material = launchParams.materialMap[id];        // mapped to enum
    auto geomNormal = viennaray::gpu::getNormal(sbtData, prd->primIDs[i]);
    auto cosTheta = __saturatef(
        -viennacore::DotProduct(prd->dir, geomNormal)); // clamp to [0,1]

    auto matParams = params->getMaterialParameters(material);

    float B_sp = matParams.B_sp;
    float Eth_sp = matParams.Eth_sp;
    float Eth_ie = matParams.Eth_ie;

    float f_sp_theta = (1.f + B_sp * (1.f - cosTheta * cosTheta)) * cosTheta;

    float sqrtE = sqrtf(prd->energy);
    float Y_sp = max(sqrtE - Eth_sp, 0.f) * f_sp_theta;
    float Y_ie = max(sqrtE - Eth_ie, 0.f) * cosTheta;

    material = static_cast<int>(Material::Polymer);
    matParams = params->getMaterialParameters(material);
    float Y_pe = max(sqrtE - matParams.Eth_ie, 0.f) * cosTheta;

    // ionSputterFlux
    atomicAdd(&launchParams
                   .resultBuffer[viennaray::gpu::getIdxOffset(0, launchParams) +
                                 prd->primIDs[i]],
              static_cast<viennaray::gpu::ResultType>(Y_sp));
    // ionEnhancedFlux
    atomicAdd(&launchParams
                   .resultBuffer[viennaray::gpu::getIdxOffset(1, launchParams) +
                                 prd->primIDs[i]],
              static_cast<viennaray::gpu::ResultType>(Y_ie));
    // ionpeFlux
    atomicAdd(&launchParams
                   .resultBuffer[viennaray::gpu::getIdxOffset(2, launchParams) +
                                 prd->primIDs[i]],
              static_cast<viennaray::gpu::ResultType>(Y_pe));
  }
}

__forceinline__ __device__ void
fluorocarbonIonReflection(const void *sbtData,
                          viennaray::gpu::PerRayData *prd) {
  viennaps::gpu::FluorocarbonParameters *params =
      reinterpret_cast<viennaps::gpu::FluorocarbonParameters *>(
          launchParams.customData);
  auto geomNormal = viennaray::gpu::getNormal(sbtData, prd->primID);
  auto cosTheta = __saturatef(
      -viennacore::DotProduct(prd->dir, geomNormal)); // clamp to [0,1]
  float angle = acosf(cosTheta);

  viennaps::impl::updateEnergy(prd, params->Ions.inflectAngle, params->Ions.n_l,
                               angle);

  if (prd->energy > params->Ions.minEnergy) {
    viennaray::gpu::conedCosineReflection(
        prd, geomNormal, M_PI_2f - min(angle, params->Ions.minAngle));
    // continue tracing the particle
  } else {
    prd->rayWeight = 0.f; // terminate particle
  }
}

__forceinline__ __device__ void
fluorocarbonIonInit(viennaray::gpu::PerRayData *prd) {
  using namespace viennaps;
  gpu::FluorocarbonParameters *params =
      reinterpret_cast<gpu::FluorocarbonParameters *>(launchParams.customData);
  impl::initNormalDistEnergy(prd, params->Ions.meanEnergy,
                             params->Ions.sigmaEnergy);
}
