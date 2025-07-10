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

  auto toProcessData() const {
    std::unordered_map<std::string, std::vector<NumericType>> processData;

    processData["Plane Wafer Rate"] = {planeWaferRate};
    for (const auto &pair : materialPlaneWaferRate) {
      processData[MaterialMap::getMaterialName(pair.first) + " PWR"] = {
          pair.second};
    }

    processData["Mean Energy"] = {meanEnergy};
    processData["Sigma Energy"] = {sigmaEnergy};
    processData["Threshold Energy"] = {thresholdEnergy};
    processData["Exponent"] = {exponent};
    processData["n_l"] = {n_l};
    processData["Inflect Angle"] = {inflectAngle * M_PI / 180.};
    processData["Min Angle"] = {minAngle * M_PI / 180.};
    processData["Tilt Angle"] = {tiltAngle * M_PI / 180.};

    if (redepositionRate > 0) {
      processData["Redeposition Threshold"] = {redepositionThreshold};
      processData["Redeposition Rate"] = {redepositionRate};
    }

    return processData;
  }
};
} // namespace viennaps