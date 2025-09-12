#pragma once

namespace viennaps::gpu::impl {
struct IonParams {
  float thetaRMin = 0.f;
  float thetaRMax = 0.f;
  float minAngle = 0.f;
  float B_sp = 0.f;
  float meanEnergy = 0.f;
  float sigmaEnergy = 0.f;
  float thresholdEnergy = 0.f;
  float inflectAngle = 0.f;
  float n = 0.f;
  float yieldFac = 0.f;
};
} // namespace viennaps::gpu::impl