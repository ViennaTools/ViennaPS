#pragma once

#include "../psConstants.hpp"

#include <unordered_map>

namespace viennaps {

template <typename NumericType> struct PlasmaEtchingParameters {
  // fluxes in (1e15 /cm² /s)
  NumericType ionFlux = 12.;
  NumericType etchantFlux = 1.8e3;
  NumericType passivationFlux = 1.0e2;

  // sticking probabilities
  std::unordered_map<int, NumericType> beta_E;
  std::unordered_map<int, NumericType> beta_P;

  NumericType etchStopDepth = std::numeric_limits<NumericType>::lowest();

  // Mask
  struct MaskType {
    // density
    NumericType rho = 500.; // 1e22 atoms/cm³

    NumericType Eth_sp = 20.; // eV
    NumericType A_sp = 0.0139;
    NumericType B_sp = 9.3;
  } Mask;

  // Polymer
  struct PolymerType {
    // density
    NumericType rho = 5.0; // 1e22 atoms/cm³

    // sputtering coefficients
    NumericType Eth_sp = 15.; // eV
    NumericType A_sp = 0.02;
    NumericType B_sp = 8.5;
  } Polymer;

  // Etching material
  struct MaterialType {
    // density
    NumericType rho = 5.02; // 1e22 atoms/cm³

    // sputtering coefficients
    NumericType Eth_sp = 20.; // eV
    NumericType Eth_ie = 15.; // eV

    NumericType A_sp = 0.0337;
    NumericType B_sp = 9.3;
    // unused
    // NumericType theta_g_sp = M_PI_2; // angle where yield is zero [rad]

    NumericType A_ie = 7.;
    NumericType B_ie = 0.8;
    // unused
    // NumericType theta_g_ie =
    //     constants::degToRad(78); // angle where yield is zero [rad]

    // chemical etching
    NumericType k_sigma = 3.0e2;     // in (1e15 cm⁻²s⁻¹)
    NumericType beta_sigma = 4.0e-2; // in (1e15 cm⁻²s⁻¹)
  } Substrate;

  // Passivation
  struct PassivationType {
    // sputtering coefficients
    NumericType Eth_ie = 10.; // eV
    NumericType A_ie = 3;
  } Passivation;

  struct IonType {
    NumericType meanEnergy = 100.; // eV
    NumericType sigmaEnergy = 10.; // eV
    NumericType exponent = 500.;

    NumericType inflectAngle = 1.55334303;
    NumericType n_l = 10.;
    NumericType minAngle = 1.3962634;

    NumericType thetaRMin = constants::degToRad(70.);
    NumericType thetaRMax = constants::degToRad(90.);
  } Ions;

  auto toProcessMetaData() const {
    std::unordered_map<std::string, std::vector<double>> processData;

    processData["Ion Flux"] = {ionFlux};
    processData["Etchant Flux"] = {etchantFlux};
    processData["Passivation Flux"] = {passivationFlux};

    for (const auto &pair : beta_E) {
      processData["Beta_E " + std::to_string(pair.first)] = {pair.second};
    }
    for (const auto &pair : beta_P) {
      processData["Beta_P " + std::to_string(pair.first)] = {pair.second};
    }

    if (etchStopDepth != std::numeric_limits<NumericType>::lowest())
      processData["Etch Stop Depth"] = {etchStopDepth};

    // Mask
    processData["Mask Rho"] = {Mask.rho};
    processData["Mask Eth_sp"] = {Mask.Eth_sp};
    processData["Mask A_sp"] = {Mask.A_sp};
    processData["Mask B_sp"] = {Mask.B_sp};

    // Polymer
    processData["Polymer Rho"] = {Polymer.rho};
    processData["Polymer Eth_sp"] = {Polymer.Eth_sp};
    processData["Polymer A_sp"] = {Polymer.A_sp};
    processData["Polymer B_sp"] = {Polymer.B_sp};

    // Material
    processData["Substrate Rho"] = {Substrate.rho};
    processData["Substrate Eth_sp"] = {Substrate.Eth_sp};
    processData["Substrate Eth_ie"] = {Substrate.Eth_ie};
    processData["Substrate A_sp"] = {Substrate.A_sp};
    processData["Substrate B_sp"] = {Substrate.B_sp};
    // processData["Substrate Theta G Sp"] = {Substrate.theta_g_sp};
    processData["Substrate A_ie"] = {Substrate.A_ie};
    processData["Substrate B_ie"] = {Substrate.B_ie};
    // processData["Substrate Theta G Ie"] = {Substrate.theta_g_ie};
    processData["Substrate K_sigma"] = {Substrate.k_sigma};
    processData["Substrate Beta_sigma"] = {Substrate.beta_sigma};

    // Passivation
    if (passivationFlux > 0) {
      processData["Passivation Eth_ie"] = {Passivation.Eth_ie};
      processData["Passivation A_ie"] = {Passivation.A_ie};
    }

    // Ions
    if (ionFlux > 0) {
      processData["Ions MeanEnergy"] = {Ions.meanEnergy};
      processData["Ions SigmaEnergy"] = {Ions.sigmaEnergy};
      processData["Ions Exponent"] = {Ions.exponent};
      processData["Ions InflectAngle"] = {Ions.inflectAngle};
      processData["Ions n_l"] = {Ions.n_l};
      processData["Ions MinAngle"] = {Ions.minAngle};
      processData["Ions ThetaRMin"] = {Ions.thetaRMin};
      processData["Ions ThetaRMax"] = {Ions.thetaRMax};
    }

    return processData;
  }

  PlasmaEtchingParameters<float> convertToFloat() const {
    PlasmaEtchingParameters<float> pParams;
    if constexpr (std::is_same_v<NumericType, float>) {
      pParams = *this;
    } else {
      pParams.ionFlux = static_cast<float>(ionFlux);
      pParams.etchantFlux = static_cast<float>(etchantFlux);
      pParams.passivationFlux = static_cast<float>(passivationFlux);

      for (auto &pair : beta_E) {
        pParams.beta_E[pair.first] = static_cast<float>(pair.second);
      }
      for (auto &pair : beta_P) {
        pParams.beta_P[pair.first] = static_cast<float>(pair.second);
      }

      pParams.etchStopDepth = static_cast<float>(etchStopDepth);

      pParams.Mask.A_sp = static_cast<float>(Mask.A_sp);
      pParams.Mask.B_sp = static_cast<float>(Mask.B_sp);
      pParams.Mask.Eth_sp = static_cast<float>(Mask.Eth_sp);
      pParams.Mask.rho = static_cast<float>(Mask.rho);

      pParams.Polymer.A_sp = static_cast<float>(Polymer.A_sp);
      pParams.Polymer.B_sp = static_cast<float>(Polymer.B_sp);
      pParams.Polymer.Eth_sp = static_cast<float>(Polymer.Eth_sp);
      pParams.Polymer.rho = static_cast<float>(Polymer.rho);

      pParams.Substrate.A_ie = static_cast<float>(Substrate.A_ie);
      pParams.Substrate.A_sp = static_cast<float>(Substrate.A_sp);
      pParams.Substrate.B_ie = static_cast<float>(Substrate.B_ie);
      pParams.Substrate.B_sp = static_cast<float>(Substrate.B_sp);
      pParams.Substrate.Eth_ie = static_cast<float>(Substrate.Eth_ie);
      pParams.Substrate.Eth_sp = static_cast<float>(Substrate.Eth_sp);
      pParams.Substrate.k_sigma = static_cast<float>(Substrate.k_sigma);
      pParams.Substrate.beta_sigma = static_cast<float>(Substrate.beta_sigma);
      pParams.Substrate.rho = static_cast<float>(Substrate.rho);

      pParams.Passivation.A_ie = static_cast<float>(Passivation.A_ie);
      pParams.Passivation.Eth_ie = static_cast<float>(Passivation.Eth_ie);

      pParams.Ions.exponent = static_cast<float>(Ions.exponent);
      pParams.Ions.meanEnergy = static_cast<float>(Ions.meanEnergy);
      pParams.Ions.sigmaEnergy = static_cast<float>(Ions.sigmaEnergy);
      pParams.Ions.inflectAngle = static_cast<float>(Ions.inflectAngle);
      pParams.Ions.n_l = static_cast<float>(Ions.n_l);
      pParams.Ions.minAngle = static_cast<float>(Ions.minAngle);
      pParams.Ions.thetaRMin = static_cast<float>(Ions.thetaRMin);
      pParams.Ions.thetaRMax = static_cast<float>(Ions.thetaRMax);
    }
    return pParams;
  }
};

} // namespace viennaps
