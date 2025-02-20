#pragma once

#include "../psConstants.hpp"
#include "../psMaterials.hpp"
#include <unordered_map>

namespace viennaps {

template <typename NumericType> struct CF4O2Parameters {
  // fluxes in (1e15 /cm² /s)
  NumericType ionFlux = 12.;
  NumericType etchantFlux = 1.8e3;
  NumericType oxygenFlux = 1.0e2;
  NumericType polymerFlux = 1.0e2;

  // sticking probabilities
  std::unordered_map<Material, NumericType> gamma_F            = {{Material::Mask, 0.7}, {Material::Si, 0.7}, {Material::SiGe, 0.7}};
  std::unordered_map<Material, NumericType> gamma_F_oxidized   = {{Material::Mask, 0.3}, {Material::Si, 0.3}, {Material::SiGe, 0.3}};
  std::unordered_map<Material, NumericType> gamma_O            = {{Material::Mask, 1.0}, {Material::Si, 1.0}, {Material::SiGe, 1.0}};
  std::unordered_map<Material, NumericType> gamma_O_passivated = {{Material::Mask, 0.3}, {Material::Si, 0.3}, {Material::SiGe, 0.3}};
  std::unordered_map<Material, NumericType> gamma_C            = {{Material::Mask, 1.0}, {Material::Si, 1.0}, {Material::SiGe, 1.0}};
  std::unordered_map<Material, NumericType> gamma_C_oxidized   = {{Material::Mask, 0.3}, {Material::Si, 0.3}, {Material::SiGe, 0.3}};

  NumericType etchStopDepth = std::numeric_limits<NumericType>::lowest();
  bool fluxIncludeSticking = false;

  // Mask
  struct MaskType {
    // density
    NumericType rho = 500.; // 1e22 atoms/cm³

    NumericType Eth_sp = 20.; // eV
    NumericType A_sp = 0.0139;
    NumericType B_sp = 9.3;
  } Mask;

  // SiGe Material Properties
  struct SiGeType {
    NumericType x = 0.3;
    NumericType rho = (5.02 - x * 0.60); // 1e22 atoms/cm³ (example value)

    // Sputtering coefficients (adjusted for SiGe)
    NumericType Eth_sp = 18.; // eV
    NumericType Eth_ie = 14.; // eV

    NumericType A_sp = 0.03;
    NumericType B_sp = 8.5;
    NumericType theta_g_sp = M_PI_2; // Angle where yield is zero [rad]

    NumericType A_ie = 6.5;
    NumericType B_ie = 0.75;
    NumericType theta_g_ie = constants::degToRad(76); // Angle where yield is zero [rad]

    // Chemical etching (separate rates for F on O)
    NumericType k_sigma = this->k_sigma_SiGe(x); // F chemical etching rate on SiGe (1e15 cm⁻²s⁻¹)
    NumericType beta_sigma = 4.0e-2; // Oxygen dissociation rate for SiGe (1e15 cm⁻²s⁻¹)

    NumericType k_sigma_SiGe(const NumericType x) const {
      return 2.982146363636e2 + 1.141883636364e2 * x;
    }
  } SiGe;

  // Si
  struct SiType {
    // density
    NumericType rho = 5.02; // 1e22 atoms/cm³

    // sputtering coefficients
    NumericType Eth_sp = 20.; // eV
    NumericType Eth_ie = 15.; // eV

    NumericType A_sp = 0.0337;
    NumericType B_sp = 9.3;
    NumericType theta_g_sp = M_PI_2; // angle where yield is zero [rad]

    NumericType A_ie = 7.;
    NumericType B_ie = 0.8;
    NumericType theta_g_ie =
        constants::degToRad(78); // angle where yield is zero [rad]

    // chemical etching
    NumericType k_sigma = 3.0e2;     // in (1e15 cm⁻²s⁻¹)
    NumericType beta_sigma = 4.0e-2; // in (1e15 cm⁻²s⁻¹)
    // New parameter: F etching rate on O-covered Si
    // NumericType k_sigma_F_on_O = 3.02;
  } Si;

  // Passivation
  struct PassivationType {
    // sputtering coefficients
    NumericType Eth_O_ie = 10.; // eV
    NumericType Eth_C_ie = 10.; // eV
    NumericType A_O_ie = 3;
    NumericType A_C_ie = 3;
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
