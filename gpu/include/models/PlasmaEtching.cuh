#pragma once

#include <vcContext.hpp>
#include <vcVectorType.hpp>

#include <raygLaunchParams.hpp>
#include <raygReflection.hpp>

#include <models/psPlasmaEtchingParameters.hpp>
#include <models/psgPipelineParameters.hpp>
#include <psMaterials.hpp>

extern "C" __constant__ viennaray::gpu::LaunchParams launchParams;

//
// --- Neutral particle
//

__forceinline__ __device__ void
plasmaNeutralCollision(viennaray::gpu::PerRayData *prd) {
  for (int i = 0; i < prd->ISCount; ++i) {
    atomicAdd(&launchParams.resultBuffer[getIdxOffset(0, launchParams) +
                                         prd->primIDs[i]],
              prd->rayWeight);
  }
}

__forceinline__ __device__ void
plasmaNeutralReflection(const void *sbtData, viennaray::gpu::PerRayData *prd) {
  const HitSBTDataBase *baseData =
      reinterpret_cast<const HitSBTDataBase *>(sbtData);
  float *data = (float *)baseData->cellData;
  const auto &phi_E = data[prd->primID];
  const auto &phi_P = data[prd->primID + launchParams.numElements];
  int material = launchParams.materialIds[prd->primID];
  float sticking = launchParams.materialSticking[material];
  float Seff = sticking * max(1.f - phi_E - phi_P, 0.f);
  prd->rayWeight -= prd->rayWeight * Seff;
  auto geoNormal = computeNormal(sbtData, prd->primID);
  diffuseReflection(prd, geoNormal, launchParams.D);
}

//
// --- Ion particle
//

__forceinline__ __device__ void
plasmaIonCollision(const void *sbtData, viennaray::gpu::PerRayData *prd) {
  viennaps::PlasmaEtchingParameters<float> *params =
      reinterpret_cast<viennaps::PlasmaEtchingParameters<float> *>(
          launchParams.customData);
  for (int i = 0; i < prd->ISCount; ++i) {
    int material = launchParams.materialIds[prd->primIDs[i]];
    auto geomNormal = computeNormal(sbtData, prd->primIDs[i]);
    auto cosTheta = -viennacore::DotProduct(prd->dir, geomNormal);
    float angle = acosf(max(min(cosTheta, 1.f), 0.f));

    float A_sp = params->Substrate.A_sp;
    float B_sp = params->Substrate.B_sp;
    float Eth_sp = params->Substrate.Eth_sp;
    if (categoryOf(static_cast<viennaps::Material>(material)) ==
        viennaps::MaterialCategory::Hardmask) {
      // mask
      A_sp = params->Mask.A_sp;
      B_sp = params->Mask.B_sp;
      Eth_sp = params->Mask.Eth_sp;
    } else if (static_cast<viennaps::Material>(material) ==
               viennaps::Material::Polymer) {
      // polymer
      A_sp = params->Polymer.A_sp;
      B_sp = params->Polymer.B_sp;
      Eth_sp = params->Polymer.Eth_sp;
    }

    float f_sp_theta =
        max((1.f + B_sp * (1.f - cosTheta * cosTheta)) * cosTheta, 0.f);

    float f_ie_theta = 1.f;
    if (cosTheta < 0.5f)
      f_ie_theta = max(3.f - 6.f * angle / M_PIf, 0.f);

    float sqrtE = sqrtf(prd->energy);
    float Y_sp = A_sp * max(sqrtE - sqrtf(Eth_sp), 0.f) * f_sp_theta;
    float Y_Si = params->Substrate.A_ie *
                 max(sqrtE - sqrtf(params->Substrate.Eth_ie), 0.f) * f_ie_theta;
    float Y_P = params->Passivation.A_ie *
                max(sqrtE - sqrtf(params->Passivation.Eth_ie), 0.f) *
                f_ie_theta;

    atomicAdd(&launchParams.resultBuffer[getIdxOffset(0, launchParams) +
                                         prd->primIDs[i]],
              Y_sp * prd->rayWeight);
    atomicAdd(&launchParams.resultBuffer[getIdxOffset(1, launchParams) +
                                         prd->primIDs[i]],
              Y_Si * prd->rayWeight);
    atomicAdd(&launchParams.resultBuffer[getIdxOffset(2, launchParams) +
                                         prd->primIDs[i]],
              Y_P * prd->rayWeight);
  }
}

__forceinline__ __device__ void
plasmaIonReflection(const void *sbtData, viennaray::gpu::PerRayData *prd) {
  viennaps::PlasmaEtchingParameters<float> *params =
      reinterpret_cast<viennaps::PlasmaEtchingParameters<float> *>(
          launchParams.customData);
  auto geomNormal = computeNormal(sbtData, prd->primID);
  auto cosTheta = __saturatef(
      -viennacore::DotProduct(prd->dir, geomNormal)); // clamp to [0,1]
  float angle = acosf(cosTheta);

  float sticking = 1.f;
  if (angle > params->Ions.thetaRMin) {
    sticking =
        1.f - __saturatef((angle - params->Ions.thetaRMin) /
                          (params->Ions.thetaRMax - params->Ions.thetaRMin));
  }
  prd->rayWeight -= prd->rayWeight * sticking;

  if (prd->rayWeight < launchParams.rayWeightThreshold) {
    return;
  }

  viennaps::gpu::impl::updateEnergy(prd, params->Ions.inflectAngle,
                                    params->Ions.n_l, angle);

  // Set the flag to stop tracing if the energy is below the threshold
  float minEnergy = min(params->Substrate.Eth_ie, params->Substrate.Eth_sp);
  if (prd->energy > minEnergy) {
    conedCosineReflection(prd, geomNormal,
                          M_PI_2f - min(angle, params->Ions.minAngle),
                          launchParams.D);
  } else {
    prd->rayWeight = 0.f; // terminate particle
  }
}

__forceinline__ __device__ void plasmaIonInit(viennaray::gpu::PerRayData *prd) {
  viennaps::PlasmaEtchingParameters<float> *params =
      reinterpret_cast<viennaps::PlasmaEtchingParameters<float> *>(
          launchParams.customData);

  float minEnergy = min(params->Substrate.Eth_ie, params->Substrate.Eth_sp);
  viennaps::gpu::impl::initNormalDistEnergy(
      prd, params->Ions.meanEnergy, params->Ions.sigmaEnergy, minEnergy);
}