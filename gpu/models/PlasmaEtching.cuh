#pragma once

#include "vcContext.hpp"
#include "vcVectorType.hpp"

#include "raygLaunchParams.hpp"
#include "raygReflection.hpp"

#include "materials/psMaterialMap.hpp"
#include "models/psPipelineParameters.hpp"
#include "models/psPlasmaEtchingParameters.hpp"

extern "C" __constant__ viennaray::gpu::LaunchParams launchParams;

//
// --- Neutral particle
//

__forceinline__ __device__ void
plasmaNeutralCollision(viennaray::gpu::PerRayData *prd) {
  for (int i = 0; i < prd->ISCount; ++i) {
    atomicAdd(&launchParams
                   .resultBuffer[viennaray::gpu::getIdxOffset(0, launchParams) +
                                 prd->primIDs[i]],
              (viennaray::gpu::ResultType)prd->rayWeight);
  }
}

__forceinline__ __device__ void
plasmaNeutralReflection(const void *sbtData, viennaray::gpu::PerRayData *prd) {
  const viennaray::gpu::HitSBTDataBase *baseData =
      reinterpret_cast<const viennaray::gpu::HitSBTDataBase *>(sbtData);
  float *coverages = (float *)baseData->cellData;
  const auto &phi_E = coverages[prd->primID];
  const auto &phi_P = coverages[prd->primID + launchParams.numElements];
  int id = launchParams.materialIds[prd->primID]; // consecutive ID, not enum
  float sticking = launchParams.materialSticking[id];
  float Seff = sticking * max(1.f - phi_E - phi_P, 0.f);
  prd->rayWeight -= prd->rayWeight * Seff;
  auto geoNormal = viennaray::gpu::getNormal(sbtData, prd->primID);
  viennaray::gpu::diffuseReflection(prd, geoNormal);
}

// Specialized neutral reflection for models without passivation (e.g., SF6C4F8)
__forceinline__ __device__ void
plasmaNeutralReflectionNoPassivation(const void *sbtData,
                                     viennaray::gpu::PerRayData *prd) {
  const viennaray::gpu::HitSBTDataBase *baseData =
      reinterpret_cast<const viennaray::gpu::HitSBTDataBase *>(sbtData);
  float *coverages = (float *)baseData->cellData;
  const auto &phi_E = coverages[prd->primID];
  int material = launchParams.materialIds[prd->primID];
  float sticking = launchParams.materialSticking[material];
  float Seff = sticking * max(1.f - phi_E, 0.f);
  prd->rayWeight -= prd->rayWeight * Seff;
  auto geoNormal = viennaray::gpu::getNormal(sbtData, prd->primID);
  viennaray::gpu::diffuseReflection(prd, geoNormal);
}

//
// --- Ion particle
//

__forceinline__ __device__ void
plasmaIonCollision(const void *sbtData, viennaray::gpu::PerRayData *prd) {
  using namespace viennaps;
  PlasmaEtchingParametersGPU *params =
      reinterpret_cast<PlasmaEtchingParametersGPU *>(launchParams.customData);
  for (int i = 0; i < prd->ISCount; ++i) {
    int id = launchParams.materialIds[prd->primIDs[i]]; // consecutive ID
    int material = launchParams.materialMap[id];        // mapped to enum
    auto geomNormal = viennaray::gpu::getNormal(sbtData, prd->primIDs[i]);
    auto cosTheta = __saturatef(
        -viennacore::DotProduct(prd->dir, geomNormal)); // clamp to [0,1]
    float angle = acosf(cosTheta);

    float A_sp = params->Substrate.A_sp;
    float B_sp = params->Substrate.B_sp;
    float Eth_sp = params->Substrate.Eth_sp;
    if (static_cast<BuiltInMaterial>(material) == BuiltInMaterial::Mask) {
      A_sp = params->Mask.A_sp;
      B_sp = params->Mask.B_sp;
      Eth_sp = params->Mask.Eth_sp;
    } else if (static_cast<BuiltInMaterial>(material) ==
               BuiltInMaterial::Polymer) {
      A_sp = params->Polymer.A_sp;
      B_sp = params->Polymer.B_sp;
      Eth_sp = params->Polymer.Eth_sp;
    }

    float f_sp_theta;
    if (static_cast<BuiltInMaterial>(material) == BuiltInMaterial::Polymer &&
        params->Polymer.usePolyCosThetaYield) {
      const float c = cosTheta;
      const float sum = params->Polymer.a1 + params->Polymer.a2 +
                        params->Polymer.a3 + params->Polymer.a4;
      f_sp_theta = (params->Polymer.a1 * c + params->Polymer.a2 * c * c +
                    params->Polymer.a3 * c * c * c +
                    params->Polymer.a4 * c * c * c * c) /
                   sum;
      f_sp_theta = max(f_sp_theta, 0.f);
    } else {
      f_sp_theta =
          max((1.f + B_sp * (1.f - cosTheta * cosTheta)) * cosTheta, 0.f);
    }

    float f_ie_theta = 1.f;
    if (cosTheta < 0.5f)
      f_ie_theta = max(3.f - 6.f * angle / M_PIf, 0.f);

    float sqrtE = sqrtf(prd->energy);
    float Y_sp = A_sp * max(sqrtE - Eth_sp, 0.f) * f_sp_theta;
    float Y_Si = params->Substrate.A_ie *
                 max(sqrtE - params->Substrate.Eth_ie, 0.f) * f_ie_theta;
    float Y_P = params->Passivation.A_ie *
                max(sqrtE - params->Passivation.Eth_ie, 0.f) * f_ie_theta;

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
plasmaIonReflection(const void *sbtData, viennaray::gpu::PerRayData *prd) {
  viennaps::PlasmaEtchingParametersGPU *params =
      reinterpret_cast<viennaps::PlasmaEtchingParametersGPU *>(
          launchParams.customData);
  auto geomNormal = viennaray::gpu::getNormal(sbtData, prd->primID);
  auto cosTheta = __saturatef(
      -viennacore::DotProduct(prd->dir, geomNormal)); // clamp to [0,1]
  float angle = acosf(cosTheta);

  float sticking = 1.f;
  if (angle > params->Ions.thetaRMin) {
    sticking =
        1.f - __saturatef((angle - params->Ions.thetaRMin) /
                          (params->Ions.thetaRMax - params->Ions.thetaRMin));
  }

  if (sticking >= 1.f) {
    prd->rayWeight = 0.f;
    return;
  }

  viennaps::gpu::impl::updateEnergy(prd, params->Ions.inflectAngle,
                                    params->Ions.n_l, angle);

  float minEnergy = min(params->Substrate.Eth_ie, params->Substrate.Eth_sp);
  if (prd->energy > minEnergy) {
    prd->rayWeight -= prd->rayWeight * sticking;
    viennaray::gpu::conedCosineReflection(
        prd, geomNormal, M_PI_2f - min(angle, params->Ions.minAngle));
  } else {
    prd->rayWeight = 0.f; // terminate particle
  }
}

__forceinline__ __device__ void plasmaIonInit(viennaray::gpu::PerRayData *prd) {
  viennaps::PlasmaEtchingParametersGPU *params =
      reinterpret_cast<viennaps::PlasmaEtchingParametersGPU *>(
          launchParams.customData);
  viennaps::gpu::impl::initNormalDistEnergy(prd, params->Ions.meanEnergy,
                                            params->Ions.sigmaEnergy);
}
