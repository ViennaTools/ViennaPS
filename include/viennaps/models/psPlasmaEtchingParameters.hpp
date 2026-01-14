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

    NumericType inflectAngle = 1.55334303; // in rad
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
      processData["Ion MeanEnergy"] = {Ions.meanEnergy};
      processData["Ion SigmaEnergy"] = {Ions.sigmaEnergy};
      processData["Ion Exponent"] = {Ions.exponent};
      processData["Ion InflectAngle"] = {Ions.inflectAngle};
      processData["Ion n_l"] = {Ions.n_l};
      processData["Ion MinAngle"] = {Ions.minAngle};
      processData["Ion ThetaRMin"] = {Ions.thetaRMin};
      processData["Ion ThetaRMax"] = {Ions.thetaRMax};
    }

    return processData;
  }
};

#ifdef VIENNACORE_COMPILE_GPU
struct PlasmaEtchingParametersGPU {
  PlasmaEtchingParametersGPU() = default;
  template <typename NumericType>
  PlasmaEtchingParametersGPU(
      const PlasmaEtchingParameters<NumericType> &parameters) {
    Mask.A_sp = static_cast<float>(parameters.Mask.A_sp);
    Mask.B_sp = static_cast<float>(parameters.Mask.B_sp);
    Mask.Eth_sp = static_cast<float>(parameters.Mask.Eth_sp);
    Mask.rho = static_cast<float>(parameters.Mask.rho);

    Polymer.A_sp = static_cast<float>(parameters.Polymer.A_sp);
    Polymer.B_sp = static_cast<float>(parameters.Polymer.B_sp);
    Polymer.Eth_sp = static_cast<float>(parameters.Polymer.Eth_sp);
    Polymer.rho = static_cast<float>(parameters.Polymer.rho);

    Substrate.rho = static_cast<float>(parameters.Substrate.rho);
    Substrate.Eth_sp = static_cast<float>(parameters.Substrate.Eth_sp);
    Substrate.Eth_ie = static_cast<float>(parameters.Substrate.Eth_ie);
    Substrate.A_sp = static_cast<float>(parameters.Substrate.A_sp);
    Substrate.B_sp = static_cast<float>(parameters.Substrate.B_sp);
    Substrate.A_ie = static_cast<float>(parameters.Substrate.A_ie);
    Substrate.B_ie = static_cast<float>(parameters.Substrate.B_ie);
    Substrate.k_sigma = static_cast<float>(parameters.Substrate.k_sigma);
    Substrate.beta_sigma = static_cast<float>(parameters.Substrate.beta_sigma);

    Passivation.Eth_ie = static_cast<float>(parameters.Passivation.Eth_ie);
    Passivation.A_ie = static_cast<float>(parameters.Passivation.A_ie);

    Ions.meanEnergy = static_cast<float>(parameters.Ions.meanEnergy);
    Ions.sigmaEnergy = static_cast<float>(parameters.Ions.sigmaEnergy);
    Ions.exponent = static_cast<float>(parameters.Ions.exponent);
    Ions.inflectAngle = static_cast<float>(parameters.Ions.inflectAngle);
    Ions.n_l = static_cast<float>(parameters.Ions.n_l);
    Ions.minAngle = static_cast<float>(parameters.Ions.minAngle);
    Ions.thetaRMin = static_cast<float>(parameters.Ions.thetaRMin);
    Ions.thetaRMax = static_cast<float>(parameters.Ions.thetaRMax);
  }

  PlasmaEtchingParameters<float>::MaskType Mask;
  PlasmaEtchingParameters<float>::PolymerType Polymer;
  PlasmaEtchingParameters<float>::MaterialType Substrate;
  PlasmaEtchingParameters<float>::PassivationType Passivation;
  PlasmaEtchingParameters<float>::IonType Ions;
};
#endif

} // namespace viennaps
