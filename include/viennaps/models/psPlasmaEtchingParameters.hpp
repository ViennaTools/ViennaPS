#pragma once

#include "../materials/psMaterialValueMap.hpp"
#include "../psConstants.hpp"

#include <unordered_map>

namespace viennaps {

template <typename NumericType> struct PlasmaEtchingParameters {
  // fluxes in (1e15 /cm² /s)
  NumericType ionFlux = 12.;
  NumericType etchantFlux = 1.8e3;
  NumericType passivationFlux = 1.0e2;

  // sticking probabilities
  MaterialValueMap<NumericType> beta_E =
      MaterialValueMap<NumericType>::fromDefault(1.0);
  MaterialValueMap<NumericType> beta_P =
      MaterialValueMap<NumericType>::fromDefault(1.0);

  MaterialValueMap<NumericType> rateFactors =
      MaterialValueMap<NumericType>::fromDefault(
          1.0); // default to 1.0 for all materials
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

    // polynomial cosine angular yield form
    bool usePolyCosThetaYield = false;
    NumericType a1 = -0.26;
    NumericType a2 = 2.72;
    NumericType a3 = -4.3;
    NumericType a4 = 1.95;
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

    for (auto entry : beta_E) {
      processData["Beta_E " + MaterialMap::toString(entry.material)] =
          std::vector<double>{entry.value};
    }
    for (auto entry : beta_P) {
      processData["Beta_P " + MaterialMap::toString(entry.material)] =
          std::vector<double>{entry.value};
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
    processData["Polymer UsePolyCosThetaYield"] = {
        (double)Polymer.usePolyCosThetaYield};
    if (Polymer.usePolyCosThetaYield) {
      processData["Polymer a1"] = {(double)Polymer.a1};
      processData["Polymer a2"] = {(double)Polymer.a2};
      processData["Polymer a3"] = {(double)Polymer.a3};
      processData["Polymer a4"] = {(double)Polymer.a4};
    }

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
namespace gpu {
// This struct only contains the parameters that are needed for the GPU kernel.
// The other parameters are not needed on the GPU and are therefore not included
// here.
struct PlasmaEtchingParameters {

  float Mask_B_sp = 0.f;
  float Mask_Eth_sp = 0.f;

  float Polymer_B_sp = 0.f;
  float Polymer_Eth_sp = 0.f;
  float Polymer_a1 = 0.f;
  float Polymer_a2 = 0.f;
  float Polymer_a3 = 0.f;
  float Polymer_a4 = 0.f;
  float Polymer_aSum = 0.f;

  float Substrate_Eth_sp = 0.f;
  float Substrate_Eth_ie = 0.f;
  float Substrate_B_sp = 0.f;
  float Substrate_B_ie = 0.f;

  float Passivation_Eth_ie = 0.f;

  float Ions_meanEnergy = 0.f;
  float Ions_sigmaEnergy = 0.f;
  float Ions_exponent = 0.f;
  float Ions_inflectAngle = 0.f;
  float Ions_n_l = 0.f;
  float Ions_minAngle = 0.f;
  float Ions_thetaRMin = 0.f;
  float Ions_thetaRMax = 0.f;

  PlasmaEtchingParameters() = default;

  // Precompute the square roots of the threshold energies for the GPU kernel
  template <typename Parameters>
  PlasmaEtchingParameters(const Parameters &parameters) {
    Mask_B_sp = static_cast<float>(parameters.Mask.B_sp);
    Mask_Eth_sp = static_cast<float>(std::sqrt(parameters.Mask.Eth_sp));

    Polymer_B_sp = static_cast<float>(parameters.Polymer.B_sp);
    Polymer_Eth_sp = static_cast<float>(std::sqrt(parameters.Polymer.Eth_sp));
    Polymer_a1 = static_cast<float>(parameters.Polymer.a1);
    Polymer_a2 = static_cast<float>(parameters.Polymer.a2);
    Polymer_a3 = static_cast<float>(parameters.Polymer.a3);
    Polymer_a4 = static_cast<float>(parameters.Polymer.a4);
    Polymer_aSum = Polymer_a1 + Polymer_a2 + Polymer_a3 + Polymer_a4;

    Substrate_Eth_sp =
        static_cast<float>(std::sqrt(parameters.Substrate.Eth_sp));
    Substrate_Eth_ie =
        static_cast<float>(std::sqrt(parameters.Substrate.Eth_ie));
    Substrate_B_sp = static_cast<float>(parameters.Substrate.B_sp);
    Substrate_B_ie = static_cast<float>(parameters.Substrate.B_ie);

    Passivation_Eth_ie =
        static_cast<float>(std::sqrt(parameters.Passivation.Eth_ie));

    Ions_meanEnergy = static_cast<float>(parameters.Ions.meanEnergy);
    Ions_sigmaEnergy = static_cast<float>(parameters.Ions.sigmaEnergy);
    Ions_exponent = static_cast<float>(parameters.Ions.exponent);
    Ions_inflectAngle = static_cast<float>(parameters.Ions.inflectAngle);
    Ions_n_l = static_cast<float>(parameters.Ions.n_l);
    Ions_minAngle = static_cast<float>(parameters.Ions.minAngle);
    Ions_thetaRMin = static_cast<float>(parameters.Ions.thetaRMin);
    Ions_thetaRMax = static_cast<float>(parameters.Ions.thetaRMax);
  }
};
} // namespace gpu
#endif

} // namespace viennaps
