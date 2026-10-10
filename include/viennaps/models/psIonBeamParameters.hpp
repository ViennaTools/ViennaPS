#pragma once

#include "../materials/psMaterialMap.hpp"
#include "../materials/psMaterialValueMap.hpp"
#include "../psConstants.hpp"

#include <cassert>
#include <cstdint>
#include <functional>
#include <limits>
#include <type_traits>
#include <unordered_map>

namespace viennaps {
template <typename NumericType> struct IBEParameters {

  // Rates
  NumericType planeWaferRate = 1.;
  MaterialValueMap<NumericType> materialPlaneWaferRate;

  NumericType meanEnergy = 250;     // eV
  NumericType sigmaEnergy = 10;     // eV
  NumericType thresholdEnergy = 20; // eV
  NumericType n_l = 10;
  NumericType inflectAngle = 89; // degree
  NumericType minAngle = 85;     // degree
  NumericType tiltAngle = 0;     // degree
  NumericType exponent = 100;
  bool rotatingWafer = false;

  // Either use the yieldFunction or the Cos4 parameters. If cos4Yield.isDefined
  // is true, the yieldFunction will be ignored.

  // Yield function depending on incident angle theta (in rad)
  std::function<NumericType(NumericType)> yieldFunction =
      [](NumericType theta) { return 1.; };

  // Cos4 Yield function. Defined in DOI:10.1109/SISPAD62626.2024.10733316
  // equation (2)
  struct cos4YieldType {
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
    for (const auto &entry : materialPlaneWaferRate) {
      processData[MaterialMap::toString(entry.material) + " PWR"] = {
          entry.value};
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

#ifdef VIENNACORE_COMPILE_GPU

namespace gpu {
struct IonParams {
  // Angle
  float tiltAngle = 0.f; // in rad
  bool rotating = false;

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

  // Yield
  float B_sp = 0.f;

  float a1 = 0.f;
  float a2 = 0.f;
  float a3 = 0.f;
  float a4 = 0.f;
  float aSum = 0.f;

  IonParams() = default;

  template <typename Parameters> explicit IonParams(const Parameters &params) {
    tiltAngle = static_cast<float>(constants::degToRad(params.tiltAngle));
    rotating = params.rotatingWafer;

    thetaRMin = static_cast<float>(constants::degToRad(params.thetaRMin));
    thetaRMax = static_cast<float>(constants::degToRad(params.thetaRMax));

    meanEnergy = static_cast<float>(params.meanEnergy);
    sigmaEnergy = static_cast<float>(params.sigmaEnergy);
    thresholdEnergy = static_cast<float>(
        std::sqrt(params.thresholdEnergy)); // precompute sqrt

    redepositionRate = static_cast<float>(params.redepositionRate);
    redepositionThreshold = static_cast<float>(params.redepositionThreshold);

    minAngle = static_cast<float>(constants::degToRad(params.minAngle));
    inflectAngle = static_cast<float>(constants::degToRad(params.inflectAngle));
    n_l = static_cast<float>(params.n_l);

    // B_sp is not used in the IBE model, but in the MultiParticle model
    if (params.cos4Yield.isDefined) {
      a1 = static_cast<float>(params.cos4Yield.a1);
      a2 = static_cast<float>(params.cos4Yield.a2);
      a3 = static_cast<float>(params.cos4Yield.a3);
      a4 = static_cast<float>(params.cos4Yield.a4);
      aSum = static_cast<float>(params.cos4Yield.aSum());
    }
  }
};
} // namespace gpu

#endif
} // namespace viennaps