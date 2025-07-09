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
};

} // namespace viennaps
