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
  // Chemisorbed HF reacting with SiO2 (ion-assisted pathway)
  struct ReactionType {
    NumericType A_r = 3.0e2;    // pre-exponential factor (1e15 cm⁻²s⁻¹)
    NumericType E_a = 0.10;     // reaction activation energy (eV)
  } Reaction;

  // Direct (pure) chemical etching: physisorbed HF reacts with SiO2 thermally,
  // no ion activation needed. Higher barrier than chemisorption pathway.
  // Negligible at cryo (150K~200K), significant at room temperature (300K).
  struct DirectReactionType {
    NumericType A_r = 1.0e4;    // pre-exponential factor
    NumericType E_a = 0.25;     // activation energy (eV) > chemisorption E_a
  } DirectReaction;

  // Ion activation: ions convert physisorbed HF -> chemisorbed HF
  // k_act_ion = A_act * ionEnhancedFlux * ionFlux  [same units as Gamma_HF]
  struct IonActivationType {
    NumericType A_act = 1.0;    // dimensionless scaling factor
  } IonActivation;

  // Model configuration: select which physical features are active.
  // Enables progressive comparison: none → +temp → +phys → +diffusion
  struct ModelConfigType {
    bool useTemperatureDependence = true;  // Frenkel-Arrhenius/Arrhenius for rates
    bool usePhysisorption         = true;  // two-state model (theta_phys + theta_chem)
    bool useSurfaceDiffusion      = true;  // 1D diffusion PDE along surface chain
    NumericType T_ref = 300.;              // fixed reference T when !useTemperatureDependence
  } Config;

  // Surface diffusion of physisorbed HF along the surface
  // D_s(T) = D0 * exp(-omega * E_des / kB*T)
  // E_diff = omega * E_des  (corrugation ratio from Lill 2023, Table IV: omega~0.2-0.3)
  struct DiffusionType {
    NumericType D0    = 1.0e3;  // nm²/s, pre-exponential
    NumericType omega = 0.25;   // corrugation ratio
  } Diffusion;

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

  // Raw Arrhenius/Frenkel-Arrhenius at actual temperature
  NumericType k_des() const {
    return Desorption.nu0 * std::exp(-Desorption.E_des / (constants::kB * temperature));
  }
  NumericType k_r() const {
    return Reaction.A_r * std::exp(-Reaction.E_a / (constants::kB * temperature));
  }
  NumericType k_r_direct() const {
    return DirectReaction.A_r * std::exp(-DirectReaction.E_a / (constants::kB * temperature));
  }
  NumericType D_s() const {
    const NumericType E_diff = Diffusion.omega * Desorption.E_des;
    return Diffusion.D0 * std::exp(-E_diff / (constants::kB * temperature));
  }

  // Config-aware rates: use T_ref when !useTemperatureDependence
  NumericType effective_k_des() const {
    const NumericType T = Config.useTemperatureDependence ? temperature : Config.T_ref;
    return Desorption.nu0 * std::exp(-Desorption.E_des / (constants::kB * T));
  }
  NumericType effective_k_r() const {
    const NumericType T = Config.useTemperatureDependence ? temperature : Config.T_ref;
    return Reaction.A_r * std::exp(-Reaction.E_a / (constants::kB * T));
  }
  NumericType effective_k_r_direct() const {
    const NumericType T = Config.useTemperatureDependence ? temperature : Config.T_ref;
    return DirectReaction.A_r * std::exp(-DirectReaction.E_a / (constants::kB * T));
  }
};

} // namespace viennaps
