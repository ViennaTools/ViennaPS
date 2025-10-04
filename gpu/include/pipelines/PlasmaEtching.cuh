#pragma once

#include <vcContext.hpp>
#include <vcVectorType.hpp>

#include <models/psPlasmaEtchingParameters.hpp>
#include <raygReflection.hpp>

extern "C" __constant__ viennaray::gpu::LaunchParams launchParams;

//
// --- Neutral particle
//

__forceinline__ __device__ void
plasmaNeutralCollision(viennaray::gpu::PerRayData *prd) {
  for (int i = 0; i < prd->ISCount; ++i) {
    atomicAdd(&launchParams.resultBuffer[getIdxOffset(0, launchParams) +
                                         prd->TIndex[i]],
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
    int material = launchParams.materialIds[prd->TIndex[i]];
    auto geomNormal = computeNormal(sbtData, prd->TIndex[i]);
    auto cosTheta = -viennacore::DotProduct(prd->dir, geomNormal);
    float angle = acosf(max(min(cosTheta, 1.f), 0.f));

    float A_sp = params->Substrate.A_sp;
    float B_sp = params->Substrate.B_sp;
    float Eth_sp = params->Substrate.Eth_sp;
    if (material == 0) {
      // mask
      A_sp = params->Mask.A_sp;
      B_sp = params->Mask.B_sp;
      Eth_sp = params->Mask.Eth_sp;
    } else if (material == 15) {
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
                                         prd->TIndex[i]],
              Y_sp * prd->rayWeight);
    atomicAdd(&launchParams.resultBuffer[getIdxOffset(1, launchParams) +
                                         prd->TIndex[i]],
              Y_Si * prd->rayWeight);
    atomicAdd(&launchParams.resultBuffer[getIdxOffset(2, launchParams) +
                                         prd->TIndex[i]],
              Y_P * prd->rayWeight);
  }
}

__forceinline__ __device__ void
plasmaIonReflection(const void *sbtData, viennaray::gpu::PerRayData *prd) {
  viennaps::PlasmaEtchingParameters<float> *params =
      reinterpret_cast<viennaps::PlasmaEtchingParameters<float> *>(
          launchParams.customData);
  auto geomNormal = computeNormal(sbtData, prd->primID);
  auto cosTheta = -viennacore::DotProduct(prd->dir, geomNormal);
  float angle = acosf(max(min(cosTheta, 1.f), 0.f));

  // Small incident angles are reflected with the energy fraction centered at
  // 0
  float Eref_peak = 0.f;
  float A = 1.f / (1.f + params->Ions.n_l *
                             (M_PI_2f / params->Ions.inflectAngle - 1.f));
  if (angle >= params->Ions.inflectAngle) {
    Eref_peak = 1.f - (1.f - A) * (M_PI_2f - angle) /
                          (M_PI_2f - params->Ions.inflectAngle);
  } else {
    Eref_peak = A * pow(angle / params->Ions.inflectAngle, params->Ions.n_l);
  }

  // Gaussian distribution around the Eref_peak scaled by the particle energy
  float newEnergy;
  do {
    newEnergy = getNormalDistRand(&prd->RNGstate) * prd->energy * 0.1f +
                Eref_peak * prd->energy;
  } while (newEnergy > prd->energy || newEnergy < 0.f);

  float sticking = 1.f;
  if (angle > params->Ions.thetaRMin) {
    sticking =
        1.f - max(min((angle - params->Ions.thetaRMin) /
                          (params->Ions.thetaRMax - params->Ions.thetaRMin),
                      1.f),
                  0.f);
  }
  prd->rayWeight -= prd->rayWeight * sticking;

  // Set the flag to stop tracing if the energy is below the threshold
  float minEnergy = min(params->Substrate.Eth_ie, params->Substrate.Eth_sp);
  if (newEnergy > minEnergy) {
    prd->energy = newEnergy;
    conedCosineReflection(prd, geomNormal,
                          M_PI_2f - min(angle, params->Ions.minAngle),
                          launchParams.D);
  } else {
    prd->energy = 0.f; // contineRay checks for >= 0
    // prd->rayWeight = 0.f; // Maybe add this?
  }
}

__forceinline__ __device__ void plasmaIonInit(viennaray::gpu::PerRayData *prd) {
  viennaps::PlasmaEtchingParameters<float> *params =
      reinterpret_cast<viennaps::PlasmaEtchingParameters<float> *>(
          launchParams.customData);

  float minEnergy = min(params->Substrate.Eth_ie, params->Substrate.Eth_sp);
  do {
    prd->energy = getNormalDistRand(&prd->RNGstate) * params->Ions.sigmaEnergy +
                  params->Ions.meanEnergy;
  } while (prd->energy < minEnergy);
}