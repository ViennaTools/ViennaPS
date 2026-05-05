#pragma once

#include "../psConstants.hpp"
#include <limits>

namespace viennaps {

template <typename NumericType> struct HFCryoParameters {
  // Fluxes in units of (1e15 / cm² / s)
  NumericType ionFlux = 1.;
  NumericType etchantFlux = 1.0e3;

  // Substrate temperature (K)
  NumericType temperature = 200.; // cryogenic, e.g. 173-223 K

  // HF sticking probability on bare SiO2
  NumericType gamma_HF = 0.9;

  NumericType etchStopDepth = std::numeric_limits<NumericType>::lowest();

  // Frenkel-Arrhenius desorption: k_des = nu0 * exp(-E_des / kB*T)
  struct DesorptionType {
    NumericType nu0 = 1.0e13;   // attempt frequency (1/s)
    NumericType E_des = 0.25;   // desorption activation energy (eV)
  } Desorption;

  // Arrhenius reaction rate: k_r = A_r * exp(-E_a / kB*T)
  struct ReactionType {
    NumericType A_r = 3.0e2;    // pre-exponential factor (1e15 cm⁻²s⁻¹)
    NumericType E_a = 0.10;     // reaction activation energy (eV)
  } Reaction;

  // SiO2 material properties
  struct SiO2Type {
    NumericType rho = constants::SiO2::rho; // 1e22 atoms/cm³
    NumericType Eth_sp = 18.;               // sputtering threshold (eV)
    NumericType A_sp = constants::SiO2::A_sp;
    NumericType Eth_ie = 12.;               // ion-enhanced etching threshold (eV)
    NumericType A_ie = 2.0;
  } SiO2;

  // Ion properties
  struct IonType {
    NumericType meanEnergy = 100.;          // eV
    NumericType sigmaEnergy = 10.;          // eV
    NumericType exponent = 300.;            // angular distribution exponent
    NumericType inflectAngle = constants::Ion::inflectAngle;
    NumericType n_l = constants::Ion::n_l;
    NumericType minAngle = constants::Ion::minAngle;
  } Ions;

  // Compute k_des(T) in units consistent with fluxes (1/s)
  NumericType k_des() const {
    return Desorption.nu0 *
           std::exp(-Desorption.E_des / (constants::kB * temperature));
  }

  // Compute k_r(T) in units of (1e15 cm⁻²s⁻¹)
  NumericType k_r() const {
    return Reaction.A_r *
           std::exp(-Reaction.E_a / (constants::kB * temperature));
  }
};

} // namespace viennaps
