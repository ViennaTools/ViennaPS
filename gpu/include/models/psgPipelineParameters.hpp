#pragma once

namespace viennaps::gpu::impl {
struct IonParams {

  // Sticking
  float thetaRMin = 0.f;
  float thetaRMax = 0.f;

  // Energy
  float meanEnergy = 0.f;
  float sigmaEnergy = 0.f;
  float thresholdEnergy = 0.f; // sqrt(E_threshold)

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
} // namespace viennaps::gpu::impl