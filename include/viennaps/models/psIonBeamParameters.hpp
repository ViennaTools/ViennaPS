#pragma once

#include "../psMaterials.hpp"

#include <functional>
#include <unordered_map>

namespace viennaps {
template <typename NumericType> struct IBEParameters {
  NumericType planeWaferRate = 1.;
  std::unordered_map<Material, NumericType> materialPlaneWaferRate;

  NumericType meanEnergy = 250;     // eV
  NumericType sigmaEnergy = 10;     // eV
  NumericType thresholdEnergy = 20; // eV
  NumericType exponent = 100;
  NumericType n_l = 10;
  NumericType inflectAngle = 89; // degree
  NumericType minAngle = 85;     // degree
  NumericType tiltAngle = 0;     // degree
  std::function<NumericType(NumericType)> yieldFunction =
      [](NumericType theta) { return 1.; };

  // Redeposition
  NumericType redepositionThreshold = 0.1;
  NumericType redepositionRate = 0.0;
};
} // namespace viennaps