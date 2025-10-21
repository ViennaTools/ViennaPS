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
  std::unordered_map<Material, NumericType> gamma_F = {
      {Material::Mask, 0.7}, {Material::Si, 0.7}, {Material::SiGe, 0.7}};
  std::unordered_map<Material, NumericType> gamma_F_oxidized = {
      {Material::Mask, 0.3}, {Material::Si, 0.3}, {Material::SiGe, 0.3}};
  std::unordered_map<Material, NumericType> gamma_O = {
      {Material::Mask, 1.0}, {Material::Si, 1.0}, {Material::SiGe, 1.0}};
  std::unordered_map<Material, NumericType> gamma_O_passivated = {
      {Material::Mask, 0.3}, {Material::Si, 0.3}, {Material::SiGe, 0.3}};
  std::unordered_map<Material, NumericType> gamma_C = {
      {Material::Mask, 1.0}, {Material::Si, 1.0}, {Material::SiGe, 1.0}};
  std::unordered_map<Material, NumericType> gamma_C_oxidized = {
      {Material::Mask, 0.3}, {Material::Si, 0.3}, {Material::SiGe, 0.3}};

  NumericType etchStopDepth = std::numeric_limits<NumericType>::lowest();
  bool fluxIncludeSticking = false;

  // Mask
  struct MaskType {
    // density
    NumericType rho = 500.; // 1e22 atoms/cm³

    NumericType Eth_sp = 20.; // eV
    NumericType A_sp = 0.0139;
  } Mask;

  // SiGe Material Properties
  struct SiGeType {
    NumericType x = 0.3;
    NumericType rho = (5.02 - x * 0.60); // 1e22 atoms/cm³ (example value)

    // Sputtering coefficients (adjusted for SiGe)
    NumericType Eth_sp = 18.; // eV
    NumericType Eth_ie = 14.; // eV

    NumericType A_sp = 0.03;
    NumericType A_ie = 6.5;

    // Chemical etching (separate rates for F on O)
    NumericType k_sigma =
        this->k_sigma_SiGe(x); // F chemical etching rate on SiGe (1e15 cm⁻²s⁻¹)
    NumericType beta_sigma =
        4.0e-2; // Oxygen dissociation rate for SiGe (1e15 cm⁻²s⁻¹)

    NumericType k_sigma_SiGe(const NumericType x) const {
      return 3.00e2 * std::exp(0.4675 * x);
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
    NumericType A_ie = 7.;

    // chemical etching
    NumericType k_sigma = 3.0e2;     // in (1e15 cm⁻²s⁻¹)
    NumericType beta_sigma = 4.0e-2; // in (1e15 cm⁻²s⁻¹)
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
  } Ions;

  auto toProcessMetaData() const {
    std::unordered_map<std::string, std::vector<double>> metaData;

    // put everything into the meta data
    metaData["ionFlux"] = {ionFlux};
    metaData["etchantFlux"] = {etchantFlux};
    metaData["oxygenFlux"] = {oxygenFlux};
    metaData["polymerFlux"] = {polymerFlux};
    metaData["MaskRho"] = {Mask.rho};
    metaData["MaskEthSp"] = {Mask.Eth_sp};
    metaData["MaskASp"] = {Mask.A_sp};
    metaData["SiGeRho"] = {SiGe.rho};
    metaData["SiGeEthSp"] = {SiGe.Eth_sp};
    metaData["SiGeEthIe"] = {SiGe.Eth_ie};
    metaData["SiGeASp"] = {SiGe.A_sp};
    metaData["SiGeAIe"] = {SiGe.A_ie};
    metaData["SiGeKSigma"] = {SiGe.k_sigma};
    metaData["SiGeBetaSigma"] = {SiGe.beta_sigma};
    metaData["SiRho"] = {Si.rho};
    metaData["SiEthSp"] = {Si.Eth_sp};
    metaData["SiEthIe"] = {Si.Eth_ie};
    metaData["SiASp"] = {Si.A_sp};
    metaData["SiAIe"] = {Si.A_ie};
    metaData["SiKSigma"] = {Si.k_sigma};
    metaData["SiBetaSigma"] = {Si.beta_sigma};
    metaData["PassivationEthOie"] = {Passivation.Eth_O_ie};
    metaData["PassivationEthCie"] = {Passivation.Eth_C_ie};
    metaData["PassivationAOie"] = {Passivation.A_O_ie};
    metaData["PassivationACie"] = {Passivation.A_C_ie};
    metaData["IonsMeanEnergy"] = {Ions.meanEnergy};
    metaData["IonsSigmaEnergy"] = {Ions.sigmaEnergy};
    metaData["IonsExponent"] = {Ions.exponent};
    metaData["IonsInflectAngle"] = {Ions.inflectAngle};
    metaData["IonsNL"] = {Ions.n_l};
    metaData["IonsMinAngle"] = {Ions.minAngle};

    // sticking probabilities
    for (const auto &gamma : gamma_F) {
      metaData["gamma_F_" + MaterialMap::toString(gamma.first)] = {
          gamma.second};
    }
    for (const auto &gamma : gamma_F_oxidized) {
      metaData["gamma_F_oxidized_" + MaterialMap::toString(gamma.first)] = {
          gamma.second};
    }
    for (const auto &gamma : gamma_O) {
      metaData["gamma_O_" + MaterialMap::toString(gamma.first)] = {
          gamma.second};
    }
    for (const auto &gamma : gamma_O_passivated) {
      metaData["gamma_O_passivated_" + MaterialMap::toString(gamma.first)] = {
          gamma.second};
    }
    for (const auto &gamma : gamma_C) {
      metaData["gamma_C_" + MaterialMap::toString(gamma.first)] = {
          gamma.second};
    }
    for (const auto &gamma : gamma_C_oxidized) {
      metaData["gamma_C_oxidized_" + MaterialMap::toString(gamma.first)] = {
          gamma.second};
    }

    return metaData;
  }
};

} // namespace viennaps
