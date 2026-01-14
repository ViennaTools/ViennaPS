#pragma once

#include "vcContext.hpp"
#include "vcVectorType.hpp"

#include "raygLaunchParams.hpp"
#include "raygReflection.hpp"

#include <models/psPipelineParameters.hpp>

extern "C" __constant__ viennaray::gpu::LaunchParams launchParams;

//
// --- Ion particle
//

__forceinline__ __device__ void IBECollision(const void *sbtData,
                                             viennaray::gpu::PerRayData *prd) {
  viennaps::gpu::impl::IonParams *params =
      (viennaps::gpu::impl::IonParams *)launchParams.customData;
  const bool yieldDefined = abs(params->aSum) > 1e-6f;
  const bool redepositionEnabled = params->redepositionRate > 0.f;

  for (int i = 0; i < prd->ISCount; ++i) {
    auto geomNormal = viennaray::gpu::getNormal(sbtData, prd->primIDs[i]);
    auto cosTheta = __saturatef(
        -viennacore::DotProduct(prd->dir, geomNormal)); // clamp to [0,1]

    float yield = 1.f;
    if (yieldDefined) {
      float cosTheta2 = cosTheta * cosTheta;
      yield = (params->a1 * cosTheta + params->a2 * cosTheta2 +
               params->a3 * cosTheta2 * cosTheta +
               params->a4 * cosTheta2 * cosTheta2) /
              params->aSum;
    }

    // threshold energy is in sqrt scale
    yield *= max(sqrtf(prd->energy) - params->thresholdEnergy, 0.f);

    // flux array
    atomicAdd(&launchParams.resultBuffer[getIdxOffset(0, launchParams) +
                                         prd->primIDs[i]],
              (viennaray::gpu::ResultType)prd->rayWeight * yield);

    if (redepositionEnabled) {
      // redeposition array
      atomicAdd(&launchParams.resultBuffer[getIdxOffset(1, launchParams) +
                                           prd->primIDs[i]],
                (viennaray::gpu::ResultType)prd->load);
    }
  }
}

__forceinline__ __device__ void IBEReflection(const void *sbtData,
                                              viennaray::gpu::PerRayData *prd) {
  viennaps::gpu::impl::IonParams *params =
      (viennaps::gpu::impl::IonParams *)launchParams.customData;
  auto geomNormal = viennaray::gpu::getNormal(sbtData, prd->primID);
  auto cosTheta = __saturatef(
      -viennacore::DotProduct(prd->dir, geomNormal)); // clamp to [0,1]
  float theta = acosf(cosTheta);

  // Update redeposition weight
  if (params->redepositionRate > 0.f) {
    float yield = 1.f;
    if (abs(params->aSum) > 0.f) {
      float cosTheta2 = cosTheta * cosTheta;
      yield = (params->a1 * cosTheta + params->a2 * cosTheta2 +
               params->a3 * cosTheta2 * cosTheta +
               params->a4 * cosTheta2 * cosTheta2) /
              params->aSum;
    }
    yield *= max(sqrtf(prd->energy) - params->thresholdEnergy, 0.f);
    prd->load = yield;
  }

  float sticking = 1.f;
  if (theta > params->thetaRMin) {
    sticking = 1.f - __saturatef((theta - params->thetaRMin) /
                                 (params->thetaRMax - params->thetaRMin));
  }

  if (sticking >= 1.f && prd->load <= 0.f) {
    prd->rayWeight = 0.f; // terminate particle
    return;
  }

  // Update energy
  viennaps::gpu::impl::updateEnergy(prd, params->inflectAngle, params->n_l,
                                    theta);

  if (prd->energy > params->thresholdEnergy * params->thresholdEnergy ||
      prd->load > params->redepositionThreshold) {
    prd->rayWeight -= prd->rayWeight * sticking;
    viennaray::gpu::conedCosineReflection(
        prd, geomNormal, M_PI_2f - min(theta, params->minAngle),
        launchParams.D);
  } else {
    prd->rayWeight = 0.f; // terminate particle
  }
}

__forceinline__ __device__ void IBEInit(viennaray::gpu::PerRayData *prd) {
  viennaps::gpu::impl::IonParams *params =
      (viennaps::gpu::impl::IonParams *)launchParams.customData;
  viennaps::gpu::impl::initNormalDistEnergy(prd, params->meanEnergy,
                                            params->sigmaEnergy);
  prd->load = 0.f;

  if (params->rotating) {
    using namespace viennacore;

    // 1) Sample wafer rotation angle
    float4 u = curand_uniform4(&prd->RNGstate);
    float phi_stage = 2.f * M_PIf * u.x;

    // 2) Beam axis a (tilted by alpha from -z)
    float sin_phi_stage, cos_phi_stage;
    __sincosf(phi_stage, &sin_phi_stage, &cos_phi_stage);
    float sin_alpha, cos_alpha;
    __sincosf(params->tiltAngle, &sin_alpha, &cos_alpha);
    Vec3Df a{sin_alpha * cos_phi_stage, sin_alpha * sin_phi_stage,
             -cos_alpha}; // already unit length

    // 3) Build basis (e1, e2, e3)
    auto e3 = a;
    auto h = (abs(e3[2]) < 0.9) ? Vec3Df{0, 0, 1} : Vec3Df{1, 0, 0};
    auto e1 = h - e3 * DotProduct(h, e3);
    Normalize(e1);
    auto e2 = CrossProduct(e3, e1);

    // 4) Sample power-cosine around e3
    float cosTheta = powf(u.y, 1.f / (launchParams.cosineExponent + 1.f));
    float sinTheta = sqrtf(max(0.f, 1.f - cosTheta * cosTheta));
    float phi = 2.f * M_PIf * u.z;
    float sin_phi, cos_phi;
    __sincosf(phi, &sin_phi, &cos_phi);

    float lx = sinTheta * cos_phi;
    float ly = sinTheta * sin_phi;
    float lz = cosTheta;

    prd->dir = lx * e1 + ly * e2 + lz * e3;

    // 5) ensure downward (toward wafer)
    if (prd->dir[2] >= 0.f) {
      prd->dir[0] = -prd->dir[0];
      prd->dir[1] = -prd->dir[1];
      prd->dir[2] = -prd->dir[2];
    }

    if (launchParams.D == 2) {
      prd->dir[1] = prd->dir[2];
      prd->dir[2] = 0.f;
      Normalize(prd->dir);
    }
  }
}