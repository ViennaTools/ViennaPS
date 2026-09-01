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
fluorocarbonEtchantReflection(const void *sbtData,
                              viennaray::gpu::PerRayData *prd) {
  const viennaray::gpu::HitSBTDataBase *baseData =
      reinterpret_cast<const viennaray::gpu::HitSBTDataBase *>(sbtData);
  const gpu::FluorocarbonParameters *params =
      reinterpret_cast<const gpu::FluorocarbonParameters *>(
          launchParams.customData);
  float *coverages = (float *)baseData->cellData;
  const auto &phi_E = coverages[prd->primID];
  const auto &phi_P = coverages[prd->primID + launchParams.numElements];
  int id = launchParams.materialIds[prd->primID]; // consecutive ID, not enum
  float Seff = params->materials[id].beta_e * max(1.f - phi_E - phi_P, 0.f);
  prd->rayWeight -= prd->rayWeight * Seff;
  auto geoNormal = viennaray::gpu::getNormal(sbtData, prd->primID);
  viennaray::gpu::diffuseReflection(prd, geoNormal);
}

__forceinline__ __device__ void
fluorocarbonPolymerReflection(const void *sbtData,
                              viennaray::gpu::PerRayData *prd) {
  const viennaray::gpu::HitSBTDataBase *baseData =
      reinterpret_cast<const viennaray::gpu::HitSBTDataBase *>(sbtData);
  const gpu::FluorocarbonParameters *params =
      reinterpret_cast<const gpu::FluorocarbonParameters *>(
          launchParams.customData);
  float *coverages = (float *)baseData->cellData;
  const auto &phi_E = coverages[prd->primID];
  const auto &phi_P = coverages[prd->primID + launchParams.numElements];
  int id = launchParams.materialIds[prd->primID]; // consecutive ID, not enum
  float Seff = params->materials[id].beta_p * max(1.f - phi_E - phi_P, 0.f);
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

    float B_sp = params->Substrate_B_sp;
    float Eth_sp = params->Substrate_Eth_sp;
    if (static_cast<BuiltInMaterial>(material) == BuiltInMaterial::Mask) {
      B_sp = params->Mask_B_sp;
      Eth_sp = params->Mask_Eth_sp;
    } else if (static_cast<BuiltInMaterial>(material) ==
               BuiltInMaterial::Polymer) {
      B_sp = params->Polymer_B_sp;
      Eth_sp = params->Polymer_Eth_sp;
    }

    float f_sp_theta;
    if (static_cast<BuiltInMaterial>(material) == BuiltInMaterial::Polymer &&
        abs(params->Polymer_aSum) > 1e6f) {
      const float c = cosTheta;
      f_sp_theta = (params->Polymer_a1 * c + params->Polymer_a2 * c * c +
                    params->Polymer_a3 * c * c * c +
                    params->Polymer_a4 * c * c * c * c) /
                   params->Polymer_aSum;
      f_sp_theta = max(f_sp_theta, 0.f);
    } else {
      f_sp_theta =
          max((1.f + B_sp * (1.f - cosTheta * cosTheta)) * cosTheta, 0.f);
    }

    float f_ie_theta = 1.f;
    if (cosTheta < 0.5f) {
      float angle = acosf(cosTheta);
      f_ie_theta = max(3.f - 6.f * angle / M_PIf, 0.f);
    }

    float sqrtE = sqrtf(prd->energy);
    float Y_sp = max(sqrtE - Eth_sp, 0.f) * f_sp_theta;
    float Y_Si = max(sqrtE - params->Substrate_Eth_ie, 0.f) * f_ie_theta;
    float Y_P = max(sqrtE - params->Passivation_Eth_ie, 0.f) * f_ie_theta;

    atomicAdd(&launchParams
                   .resultBuffer[viennaray::gpu::getIdxOffset(0, launchParams) +
                                 prd->primIDs[i]],
              static_cast<viennaray::gpu::ResultType>(Y_sp * prd->rayWeight));
    atomicAdd(&launchParams
                   .resultBuffer[viennaray::gpu::getIdxOffset(1, launchParams) +
                                 prd->primIDs[i]],
              static_cast<viennaray::gpu::ResultType>(Y_Si * prd->rayWeight));
    atomicAdd(&launchParams
                   .resultBuffer[viennaray::gpu::getIdxOffset(2, launchParams) +
                                 prd->primIDs[i]],
              static_cast<viennaray::gpu::ResultType>(Y_P * prd->rayWeight));
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

  float sticking = 1.f;
  if (angle > params->Ions_thetaRMin) {
    sticking =
        1.f - __saturatef((angle - params->Ions_thetaRMin) /
                          (params->Ions_thetaRMax - params->Ions_thetaRMin));
  }

  if (sticking >= 1.f) {
    prd->rayWeight = 0.f;
    return;
  }

  viennaps::impl::updateEnergy(prd, params->Ions_inflectAngle, params->Ions_n_l,
                               angle);

  float minEnergy = min(params->Substrate_Eth_ie, params->Substrate_Eth_sp);
  if (prd->energy > minEnergy) {
    prd->rayWeight -= prd->rayWeight * sticking;
    viennaray::gpu::conedCosineReflection(
        prd, geomNormal, M_PI_2f - min(angle, params->Ions_minAngle));
  } else {
    prd->rayWeight = 0.f; // terminate particle
  }
}

__forceinline__ __device__ void
fluorocarbonIonInit(viennaray::gpu::PerRayData *prd) {
  viennaps::gpu::FluorocarbonParameters *params =
      reinterpret_cast<viennaps::gpu::FluorocarbonParameters *>(
          launchParams.customData);
  viennaps::impl::initNormalDistEnergy(prd, params->Ions_meanEnergy,
                                       params->Ions_sigmaEnergy);
}
