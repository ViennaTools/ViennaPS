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
__forceinline__ __device__ void updateEnergy(viennaray::gpu::PerRayData *prd,
                                             const float inflectAngle,
                                             const float n_l,
                                             const float incAngle /*rad*/) {
  float Eref_peak; // between 0 and 1
  float A = 1.f / (1.f + n_l * (M_PI_2f / inflectAngle - 1.f));
  if (incAngle >= inflectAngle) {
    Eref_peak =
        (1.f - (1.f - A) * (M_PI_2f - incAngle) / (M_PI_2f - inflectAngle));
  } else {
    Eref_peak = A * powf(incAngle / inflectAngle, n_l);
  }

  float newEnergy;
  do {
    newEnergy = getNormalDistRand(&prd->RNGstate) * prd->energy * 0.1f +
                Eref_peak * prd->energy;
  } while (newEnergy > prd->energy || newEnergy <= 0.f);

  prd->energy = newEnergy;
}
#endif

} // namespace viennaps::gpu::impl