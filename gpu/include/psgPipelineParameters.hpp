#pragma once

#include <raygRNG.hpp>

namespace viennaps::gpu::impl {
struct IonParams {

  // Sticking
  float thetaRMin = 0.f;
  float thetaRMax = 0.f;

  // Energy
  float meanEnergy = 0.f;
  float sigmaEnergy = 0.f;
  float thresholdEnergy = 0.f; // sqrt(E_threshold)

  // Redeposition
  float redepositionRate = 0.f;
  float redepositionThreshold = 0.1f;

  // Reflection Angular Distribution
  float minAngle = 0.f;     // in rad
  float inflectAngle = 0.f; // in rad
  float n_l = 10.f;

  // Sputter Yield
  float B_sp = 0.f;

  // Cos4 Yield
  float a1 = 0.f;
  float a2 = 0.f;
  float a3 = 0.f;
  float a4 = 0.f;
  float aSum = 0.f;
};

#ifdef __CUDACC__

__forceinline__ __device__ float norm_inv_from_cdf(float p) {
  // p in (0,1). Maps to N(0,1)
  return __saturatef(-1.4142135623730951f * erfcinvf(2.0f * p));
}

__forceinline__ __device__ float phi_from_x(float x) {
  // Φ(x) via erfc
  return 0.5f * erfcf(-x * 0.7071067811865475f);
}

__forceinline__ __device__ void updateEnergy(viennaray::gpu::PerRayData *prd,
                                             const float inflectAngle,
                                             const float n_l,
                                             const float incAngle /*rad*/) {
  float Eref_peak; // between 0 and 1
  const float A = 1.f / (1.f + n_l * (M_PI_2f / inflectAngle - 1.f));
  if (incAngle >= inflectAngle) {
    Eref_peak =
        (1.f - (1.f - A) * (M_PI_2f - incAngle) / (M_PI_2f - inflectAngle));
  } else {
    Eref_peak = A * powf(incAngle / inflectAngle, n_l);
  }

  const float sigma = 0.1f;
  const float a = (0.f - Eref_peak) / sigma;
  const float b = (1.f - Eref_peak) / sigma;

  const float Fa = phi_from_x(a);
  const float Fb = phi_from_x(b);
  const float width = Fb - Fa;

  // Guard extreme tails to avoid Fb==Fa
  if (width <= 1e-7f) {
    // pick midpoint in CDF space
    const float pmid = Fa + 0.5f * width;
    prd->energy *= Eref_peak + sigma * norm_inv_from_cdf(pmid);
  } else {
    const float u = curand_uniform(&prd->RNGstate); // (0,1)
    const float p = fmaf(width, u, Fa);             // Fa + u*(Fb-Fa)
    const float z = norm_inv_from_cdf(p);           // z in (a,b)
    prd->energy *= Eref_peak + sigma * z;           // ∈ (0,1)
  }
}

__forceinline__ __device__ void
initNormalDistEnergy(viennaray::gpu::PerRayData *prd, const float mean,
                     const float sigma, const float threshold) {
  const float a = (threshold - mean) / sigma;
  // Phi(a)
  const float Phi_a = phi_from_x(a);

  // If threshold <= mean, the truncation is weak.
  const float u = curand_uniform(&prd->RNGstate); // (0,1)
  const float up = fmaf(1.0f - Phi_a, u, Phi_a);  // Phi(a) + u*(1-Phi(a))
  const float z = norm_inv_from_cdf(up);          // z >= a by construction
  prd->energy = mean + sigma * z;
}
#endif

} // namespace viennaps::gpu::impl