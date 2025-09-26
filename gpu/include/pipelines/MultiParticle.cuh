#pragma once

#include <vcContext.hpp>
#include <vcVectorType.hpp>

#include <raygReflection.hpp>

extern "C" __constant__ viennaray::gpu::LaunchParams launchParams;

//
// --- Neutral particle
//

__forceinline__ __device__ void
multiNeutralCollision(viennaray::gpu::PerRayData *prd) {
  for (int i = 0; i < prd->ISCount; ++i) {
    atomicAdd(&launchParams.resultBuffer[getIdxOffset(0, launchParams) +
                                         prd->TIndex[i]],
              prd->rayWeight);
  }
}

__forceinline__ __device__ void
multiNeutralReflection(const viennaray::gpu::HitSBTData *sbtData,
                       viennaray::gpu::PerRayData *prd) {
  int material = launchParams.materialIds[prd->primID];
  float sticking = launchParams.materialSticking[material];
  prd->rayWeight -= prd->rayWeight * sticking;
  auto geoNormal = computeNormal(sbtData, prd->primID);
  diffuseReflection(prd, geoNormal, launchParams.D);
}

//
// --- Ion particle
//

__forceinline__ __device__ void
multiIonCollision(const viennaray::gpu::HitSBTData *sbtData,
                  viennaray::gpu::PerRayData *prd) {
  viennaps::gpu::impl::IonParams *params =
      (viennaps::gpu::impl::IonParams *)launchParams.customData[0];
  for (int i = 0; i < prd->ISCount; ++i) {
    auto geomNormal = computeNormal(sbtData, prd->TIndex[i]);
    auto cosTheta = -viennacore::DotProduct(prd->dir, geomNormal);
    float incomingAngle = acosf(max(min(cosTheta, 1.f), 0.f));

    float flux = prd->rayWeight;
    if (params->B_sp >= 0.f) {
      flux *= (1 + params->B_sp * (1.f - cosTheta * cosTheta)) * cosTheta;
    }

    if (params->meanEnergy > 0.f) {
      flux *= max(sqrtf(prd->energy) - sqrtf(params->thresholdEnergy), 0.f);
    }

    atomicAdd(&launchParams.resultBuffer[getIdxOffset(0, launchParams) +
                                         prd->TIndex[i]],
              flux);
  }
}

__forceinline__ __device__ void
multiIonReflection(const viennaray::gpu::HitSBTData *sbtData,
                   viennaray::gpu::PerRayData *prd) {
  viennaps::gpu::impl::IonParams *params =
      (viennaps::gpu::impl::IonParams *)launchParams.customData[0];
  auto geomNormal = computeNormal(sbtData, prd->primID);
  auto cosTheta = -viennacore::DotProduct(prd->dir, geomNormal);
  float incomingAngle = acosf(max(min(cosTheta, 1.f), 0.f));

  float sticking = 1.f;
  if (incomingAngle > params->thetaRMin)
    sticking = 1.f - min((incomingAngle - params->thetaRMin) /
                             (params->thetaRMax - params->thetaRMin),
                         1.f);
  prd->rayWeight -= prd->rayWeight * sticking;

  if (params->meanEnergy > 0.f) {
    float Eref_peak;
    float A = 1.f / (1.f + params->n * (M_PI_2f / params->inflectAngle - 1.f));
    if (incomingAngle >= params->inflectAngle) {
      Eref_peak = (1 - (1 - A) * (M_PI_2f - incomingAngle) /
                           (M_PI_2f - params->inflectAngle));
    } else {
      Eref_peak = A * powf(incomingAngle / params->inflectAngle, params->n);
    }

    float newEnergy;
    do {
      newEnergy = getNormalDistRand(&prd->RNGstate) * prd->energy * 0.1f +
                  Eref_peak * prd->energy;
    } while (newEnergy > prd->energy || newEnergy <= 0.f);

    prd->energy = newEnergy;
  }

  conedCosineReflectionNew(prd, geomNormal,
                           M_PI_2f - min(incomingAngle, params->minAngle),
                           launchParams.D);
}

__forceinline__ __device__ void multiIonInit(viennaray::gpu::PerRayData *prd) {
  viennaps::gpu::impl::IonParams *params =
      (viennaps::gpu::impl::IonParams *)launchParams.customData[0];

  if (params->meanEnergy > 0.f) {
    do {
      prd->energy = getNormalDistRand(&prd->RNGstate) * params->sigmaEnergy +
                    params->meanEnergy;
    } while (prd->energy <= 0.f);
  } else {
    prd->energy = std::numeric_limits<float>::max();
  }
}