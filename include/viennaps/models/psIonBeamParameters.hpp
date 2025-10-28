#pragma once

#include "../psMaterials.hpp"

#include <functional>
#include <unordered_map>

namespace viennaps {
template <typename NumericType> struct IBEParameters {

  // Rates
  NumericType planeWaferRate = 1.;
  std::unordered_map<Material, NumericType> materialPlaneWaferRate;

  NumericType meanEnergy = 250;     // eV
  NumericType sigmaEnergy = 10;     // eV
  NumericType thresholdEnergy = 20; // eV
  NumericType n_l = 10;
  NumericType inflectAngle = 89; // degree
  NumericType minAngle = 85;     // degree
  NumericType tiltAngle = 0;     // degree
  NumericType exponent = 100;

  // Either use the yieldFunction or the Cos4 parameters. If cos4Yield.isDefined
  // is true, the yieldFunction will be ignored.

  // Yield function depending on incident angle theta (in rad)
  std::function<NumericType(NumericType)> yieldFunction =
      [](NumericType theta) { return 1.; };

  // Cos4 Yield function. Defined in DOI:10.1109/SISPAD62626.2024.10733316
  // equation (2)
  struct {
    NumericType a1 = 0, a2 = 0, a3 = 0, a4 = 0;
    bool isDefined = false;
    NumericType aSum() const { return a1 + a2 + a3 + a4; }
  } cos4Yield;

  // Sticking
  NumericType thetaRMin = 70; // degree
  NumericType thetaRMax = 90; // degree

  // Redeposition
  NumericType redepositionThreshold = 0.1;
  NumericType redepositionRate = 0.0;

  auto toProcessMetaData() const {
    std::unordered_map<std::string, std::vector<double>> processData;

    processData["Default PWR"] = {planeWaferRate};
    for (const auto &pair : materialPlaneWaferRate) {
      processData[MaterialMap::toString(pair.first) + " PWR"] = {pair.second};
    }

    processData["Mean Energy"] = {meanEnergy};
    processData["Sigma Energy"] = {sigmaEnergy};
    processData["Threshold Energy"] = {thresholdEnergy};
    processData["Exponent"] = {exponent};
    processData["n_l"] = {n_l};
    processData["Inflect Angle"] = {inflectAngle};
    processData["Min Angle"] = {minAngle};
    processData["Tilt Angle"] = {tiltAngle};
    processData["Theta R Min"] = {thetaRMin};
    processData["Theta R Max"] = {thetaRMax};

    if (cos4Yield.isDefined) {
      processData["Cos4 a1"] = {cos4Yield.a1};
      processData["Cos4 a2"] = {cos4Yield.a2};
      processData["Cos4 a3"] = {cos4Yield.a3};
      processData["Cos4 a4"] = {cos4Yield.a4};
    }

    if (redepositionRate > 0) {
      processData["Redeposition Threshold"] = {redepositionThreshold};
      processData["Redeposition Rate"] = {redepositionRate};
    }

    return processData;
  }
};
} // namespace viennaps