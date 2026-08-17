#pragma once

#include "vcContext.hpp"
#include "vcVectorType.hpp"

#include "raygLaunchParams.hpp"
#include "raygReflection.hpp"

#include "materials/psBuiltInMaterial.hpp"
#include "models/psCF4O2Parameters.hpp"
#include "models/psPipelineParameters.hpp"

extern "C" __constant__ viennaray::gpu::LaunchParams launchParams;

namespace viennaps::gpu::impl {

__forceinline__ __device__ float
cf4o2ClampCoverage(const float value) {
  return fminf(fmaxf(value, 0.f), 1.f);
}

__forceinline__ __device__ bool
cf4o2IsOxide(const int material) {
  const auto builtIn = static_cast<viennaps::BuiltInMaterial>(material);
  return builtIn == viennaps::BuiltInMaterial::SiO2 ||
         builtIn == viennaps::BuiltInMaterial::Mask;
}

__forceinline__ __device__ float
cf4o2Gamma(const float (&map)[kBuiltInMaterialMaxId + 1],
           const int material) {
  if (material < 0 || material > static_cast<int>(kBuiltInMaterialMaxId)) {
    return map[static_cast<std::size_t>(viennaps::BuiltInMaterial::Undefined)];
  }
  return map[static_cast<std::size_t>(material)];
}

__forceinline__ __device__ void
cf4o2NeutralCollision(viennaray::gpu::PerRayData *prd, const float Seff) {
  atomicAdd(&launchParams
                 .resultBuffer[viennaray::gpu::getIdxOffset(0, launchParams) +
                               prd->primID],
            static_cast<viennaray::gpu::ResultType>(prd->rayWeight * Seff));
}

__forceinline__ __device__ void
cf4o2DiffuseReflection(const void *sbtData, viennaray::gpu::PerRayData *prd,
                       const float Seff) {
  prd->rayWeight -= prd->rayWeight * Seff;
  auto geoNormal = viennaray::gpu::getNormal(sbtData, prd->primID);
  viennaray::gpu::diffuseReflection(prd, geoNormal);
}

} // namespace viennaps::gpu::impl

__forceinline__ __device__ void
CF4O2EtchantCollision(const void *sbtData, viennaray::gpu::PerRayData *prd) {
  auto *params =
      reinterpret_cast<viennaps::CF4O2ParametersGPU *>(launchParams.customData);
  float Seff = 1.f;
  if (params->storeAbsorbedFlux) {
    const auto *baseData =
        reinterpret_cast<const viennaray::gpu::HitSBTDataBase *>(sbtData);
    float *coverages = reinterpret_cast<float *>(baseData->cellData);
    const auto &phiF = coverages[prd->primID];
    const auto &phiO = coverages[prd->primID + launchParams.numElements];
    const auto &phiC = coverages[prd->primID + 2 * launchParams.numElements];
    const int materialIdx = launchParams.materialMap[launchParams.materialIds[prd->primID]];
    const float gammaF =
        viennaps::gpu::impl::cf4o2Gamma(params->gamma_F, materialIdx);
    const float gammaFO = viennaps::gpu::impl::cf4o2Gamma(
        params->gamma_F_oxidized, materialIdx);
    Seff = gammaF * fmaxf(1.f - phiF - phiO - phiC, 0.f) + gammaFO * phiO;
  }
  viennaps::gpu::impl::cf4o2NeutralCollision(prd, Seff);
}

__forceinline__ __device__ void
CF4O2EtchantReflection(const void *sbtData, viennaray::gpu::PerRayData *prd) {
  auto *params =
      reinterpret_cast<viennaps::CF4O2ParametersGPU *>(launchParams.customData);
  const auto *baseData =
      reinterpret_cast<const viennaray::gpu::HitSBTDataBase *>(sbtData);
  float *coverages = reinterpret_cast<float *>(baseData->cellData);
  const auto &phiF = coverages[prd->primID];
  const auto &phiO = coverages[prd->primID + launchParams.numElements];
  const auto &phiC = coverages[prd->primID + 2 * launchParams.numElements];
  const int materialIdx = launchParams.materialMap[launchParams.materialIds[prd->primID]];
  const float gammaF =
      viennaps::gpu::impl::cf4o2Gamma(params->gamma_F, materialIdx);
  const float gammaFO =
      viennaps::gpu::impl::cf4o2Gamma(params->gamma_F_oxidized, materialIdx);
  const float Seff = gammaF * fmaxf(1.f - phiF - phiO - phiC, 0.f) +
                     gammaFO * phiO;
  viennaps::gpu::impl::cf4o2DiffuseReflection(sbtData, prd, Seff);
}

__forceinline__ __device__ void
CF4O2OxygenCollision(const void *sbtData, viennaray::gpu::PerRayData *prd) {
  auto *params =
      reinterpret_cast<viennaps::CF4O2ParametersGPU *>(launchParams.customData);
  float Seff = 1.f;
  if (params->storeAbsorbedFlux) {
    const auto *baseData =
        reinterpret_cast<const viennaray::gpu::HitSBTDataBase *>(sbtData);
    float *coverages = reinterpret_cast<float *>(baseData->cellData);
    const auto &phiF = coverages[prd->primID];
    const auto &phiO = coverages[prd->primID + launchParams.numElements];
    const auto &phiC = coverages[prd->primID + 2 * launchParams.numElements];
    const int materialIdx = launchParams.materialMap[launchParams.materialIds[prd->primID]];
    const float gammaO =
        viennaps::gpu::impl::cf4o2Gamma(params->gamma_O, materialIdx);
    const float gammaOP = viennaps::gpu::impl::cf4o2Gamma(
        params->gamma_O_passivated, materialIdx);
    Seff = gammaO * fmaxf(1.f - phiO - phiF - phiC, 0.f) + gammaOP * phiC;
  }
  viennaps::gpu::impl::cf4o2NeutralCollision(prd, Seff);
}

__forceinline__ __device__ void
CF4O2OxygenReflection(const void *sbtData, viennaray::gpu::PerRayData *prd) {
  auto *params =
      reinterpret_cast<viennaps::CF4O2ParametersGPU *>(launchParams.customData);
  const auto *baseData =
      reinterpret_cast<const viennaray::gpu::HitSBTDataBase *>(sbtData);
  float *coverages = reinterpret_cast<float *>(baseData->cellData);
  const auto &phiF = coverages[prd->primID];
  const auto &phiO = coverages[prd->primID + launchParams.numElements];
  const auto &phiC = coverages[prd->primID + 2 * launchParams.numElements];
  const int materialIdx = launchParams.materialMap[launchParams.materialIds[prd->primID]];
  const float gammaO =
      viennaps::gpu::impl::cf4o2Gamma(params->gamma_O, materialIdx);
  const float gammaOP =
      viennaps::gpu::impl::cf4o2Gamma(params->gamma_O_passivated, materialIdx);
  const float Seff = gammaO * fmaxf(1.f - phiO - phiF - phiC, 0.f) +
                     gammaOP * phiC;
  viennaps::gpu::impl::cf4o2DiffuseReflection(sbtData, prd, Seff);
}

__forceinline__ __device__ void
CF4O2PolymerCollision(const void *sbtData, viennaray::gpu::PerRayData *prd) {
  auto *params =
      reinterpret_cast<viennaps::CF4O2ParametersGPU *>(launchParams.customData);
  float Seff = 1.f;
  if (params->storeAbsorbedFlux) {
    const auto *baseData =
        reinterpret_cast<const viennaray::gpu::HitSBTDataBase *>(sbtData);
    float *coverages = reinterpret_cast<float *>(baseData->cellData);
    const auto &phiF = coverages[prd->primID];
    const auto &phiO = coverages[prd->primID + launchParams.numElements];
    const auto &phiC = coverages[prd->primID + 2 * launchParams.numElements];
    const int materialIdx = launchParams.materialMap[launchParams.materialIds[prd->primID]];
    const float gammaC =
        viennaps::gpu::impl::cf4o2Gamma(params->gamma_C, materialIdx);
    const float gammaCO = viennaps::gpu::impl::cf4o2Gamma(
        params->gamma_C_oxidized, materialIdx);
    Seff = gammaC * fmaxf(1.f - phiO - phiF - phiC, 0.f) + gammaCO * phiO;
  }
  viennaps::gpu::impl::cf4o2NeutralCollision(prd, Seff);
}

__forceinline__ __device__ void
CF4O2PolymerReflection(const void *sbtData, viennaray::gpu::PerRayData *prd) {
  auto *params =
      reinterpret_cast<viennaps::CF4O2ParametersGPU *>(launchParams.customData);
  const auto *baseData =
      reinterpret_cast<const viennaray::gpu::HitSBTDataBase *>(sbtData);
  float *coverages = reinterpret_cast<float *>(baseData->cellData);
  const auto &phiF = coverages[prd->primID];
  const auto &phiO = coverages[prd->primID + launchParams.numElements];
  const auto &phiC = coverages[prd->primID + 2 * launchParams.numElements];
  const int materialIdx = launchParams.materialMap[launchParams.materialIds[prd->primID]];
  const float gammaC =
      viennaps::gpu::impl::cf4o2Gamma(params->gamma_C, materialIdx);
  const float gammaCO =
      viennaps::gpu::impl::cf4o2Gamma(params->gamma_C_oxidized, materialIdx);
  const float Seff = gammaC * fmaxf(1.f - phiO - phiF - phiC, 0.f) +
                     gammaCO * phiO;
  viennaps::gpu::impl::cf4o2DiffuseReflection(sbtData, prd, Seff);
}

__forceinline__ __device__ void
CF4O2IonCollision(const void *sbtData, viennaray::gpu::PerRayData *prd) {
  auto *params =
      reinterpret_cast<viennaps::CF4O2ParametersGPU *>(launchParams.customData);
  for (int i = 0; i < prd->ISCount; ++i) {
    const auto primID = prd->primIDs[i];
    const int mappedMaterial =
        launchParams.materialMap[launchParams.materialIds[primID]];
    auto geomNormal = viennaray::gpu::getNormal(sbtData, primID);
    auto cosTheta =
        __saturatef(-viennacore::DotProduct(prd->dir, geomNormal));
    float angle = acosf(cosTheta);

    float A_sp = params->Si.A_sp;
    float Eth_sp = params->Si.Eth_sp;
    float A_ie = params->Si.A_ie;
    float Eth_ie = params->Si.Eth_ie;
    if (static_cast<viennaps::BuiltInMaterial>(mappedMaterial) ==
        viennaps::BuiltInMaterial::SiGe) {
      A_sp = params->SiGe.A_sp;
      Eth_sp = params->SiGe.Eth_sp;
      A_ie = params->SiGe.A_ie;
      Eth_ie = params->SiGe.Eth_ie;
    }
    if (static_cast<viennaps::BuiltInMaterial>(mappedMaterial) ==
        viennaps::BuiltInMaterial::Mask) {
      A_sp = params->Mask.A_sp;
      Eth_sp = params->Mask.Eth_sp;
    }

    float f_sp_theta = 1.f;
    float f_ie_theta = 1.f;
    if (cosTheta < 0.5f) {
      f_ie_theta = fmaxf(3.f - 6.f * angle / M_PIf, 0.f);
    }

    const float sqrtE = sqrtf(prd->energy);
    const float Y_sp = A_sp * fmaxf(sqrtE - Eth_sp, 0.f) * f_sp_theta;
    const float Y_Si =
        A_ie * fmaxf(sqrtE - Eth_ie, 0.f) * f_ie_theta;
    const float Y_O = params->Passivation.A_O_ie *
                      fmaxf(sqrtE - params->Passivation.Eth_O_ie, 0.f) *
                      f_ie_theta;
    const float Y_C = params->Passivation.A_C_ie *
                      fmaxf(sqrtE - params->Passivation.Eth_C_ie, 0.f) *
                      f_ie_theta;

    atomicAdd(&launchParams
                   .resultBuffer[viennaray::gpu::getIdxOffset(0, launchParams) +
                                 primID],
              static_cast<viennaray::gpu::ResultType>(Y_sp * prd->rayWeight));
    atomicAdd(&launchParams
                   .resultBuffer[viennaray::gpu::getIdxOffset(1, launchParams) +
                                 primID],
              static_cast<viennaray::gpu::ResultType>(Y_Si * prd->rayWeight));
    atomicAdd(&launchParams
                   .resultBuffer[viennaray::gpu::getIdxOffset(2, launchParams) +
                                 primID],
              static_cast<viennaray::gpu::ResultType>(Y_O * prd->rayWeight));
    atomicAdd(&launchParams
                   .resultBuffer[viennaray::gpu::getIdxOffset(3, launchParams) +
                                 primID],
              static_cast<viennaray::gpu::ResultType>(Y_C * prd->rayWeight));
  }
}

__forceinline__ __device__ void
CF4O2IonReflection(const void *sbtData, viennaray::gpu::PerRayData *prd) {
  auto *params =
      reinterpret_cast<viennaps::CF4O2ParametersGPU *>(launchParams.customData);
  auto geomNormal = viennaray::gpu::getNormal(sbtData, prd->primID);
  auto cosTheta =
      __saturatef(-viennacore::DotProduct(prd->dir, geomNormal));
  const float angle = acosf(cosTheta);

  viennaps::gpu::impl::updateEnergy(prd, params->Ions.inflectAngle,
                                    params->Ions.n_l, angle);

  const int mappedMaterial =
      launchParams.materialMap[launchParams.materialIds[prd->primID]];
  float interactionThreshold = params->Si.Eth_ie;
  if (static_cast<viennaps::BuiltInMaterial>(mappedMaterial) ==
      viennaps::BuiltInMaterial::SiGe) {
    interactionThreshold = params->SiGe.Eth_ie;
  } else if (static_cast<viennaps::BuiltInMaterial>(mappedMaterial) ==
             viennaps::BuiltInMaterial::Mask) {
    interactionThreshold = params->Mask.Eth_sp;
  }

  // Threshold energies are square-rooted during GPU parameter initialization
  // for the yield law, while prd->energy remains in eV.
  if (prd->energy > interactionThreshold * interactionThreshold) {
    viennaray::gpu::conedCosineReflection(
        prd, geomNormal, M_PI_2f - fminf(angle, params->Ions.minAngle));
  } else {
    prd->rayWeight = 0.f;
  }
}

__forceinline__ __device__ void
CF4O2IonInit(viennaray::gpu::PerRayData *prd) {
  auto *params =
      reinterpret_cast<viennaps::CF4O2ParametersGPU *>(launchParams.customData);
  viennaps::gpu::impl::initNormalDistEnergy(prd, params->Ions.meanEnergy,
                                            params->Ions.sigmaEnergy);
}
