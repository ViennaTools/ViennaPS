#pragma once

#include "../materials/psMaterialMap.hpp"
#include "../materials/psMaterialValueMap.hpp"
#include "../psConstants.hpp"
#include <unordered_map>

namespace viennaps {

template <typename NumericType> struct CF4O2Parameters {
  // fluxes in (1e15 /cm² /s)
  NumericType ionFlux = 12.;
  NumericType etchantFlux = 1.8e3;
  NumericType oxygenFlux = 1.0e2;
  NumericType polymerFlux = 1.0e2;

  // sticking probabilities
  MaterialValueMap<NumericType> gamma_F = MaterialValueMap<NumericType>(
      std::initializer_list<std::pair<Material, NumericType>>{
          {Material::SiO2, 0.7}, {Material::Si, 0.7}, {Material::SiGe, 0.7}},
      1.0);

  MaterialValueMap<NumericType> gamma_F_oxidized =
      MaterialValueMap<NumericType>(
          std::initializer_list<std::pair<Material, NumericType>>{
              {Material::SiO2, 0.3},
              {Material::Si, 0.3},
              {Material::SiGe, 0.3}},
          1.0);

  // Effective probability for forming an oxygen-containing passivation layer
  // (e.g. SiOxFy-like / oxidized-passivated state) on the bare surface.
  MaterialValueMap<NumericType> gamma_O = MaterialValueMap<NumericType>(
      std::initializer_list<std::pair<Material, NumericType>>{
          {Material::SiO2, 1.0}, {Material::Si, 1.0}, {Material::SiGe, 1.0}},
      1.0);

  // Effective probability for oxygen interacting with a polymer-covered
  // surface. In the reduced-order model this contributes both to passivation
  // formation and polymer removal.
  MaterialValueMap<NumericType> gamma_O_passivated =
      MaterialValueMap<NumericType>(
          std::initializer_list<std::pair<Material, NumericType>>{
              {Material::SiO2, 0.3},
              {Material::Si, 0.3},
              {Material::SiGe, 0.3}},
          1.0);

  MaterialValueMap<NumericType> gamma_C = MaterialValueMap<NumericType>(
      std::initializer_list<std::pair<Material, NumericType>>{
          {Material::SiO2, 1.0}, {Material::Si, 1.0}, {Material::SiGe, 1.0}},
      1.0);

  MaterialValueMap<NumericType> gamma_C_oxidized =
      MaterialValueMap<NumericType>(
          std::initializer_list<std::pair<Material, NumericType>>{
              {Material::SiO2, 0.3},
              {Material::Si, 0.3},
              {Material::SiGe, 0.3}},
          1.0);

  NumericType etchStopDepth = std::numeric_limits<NumericType>::lowest();

  // Controls what is accumulated in the local neutral flux fields:
  // false -> incident flux at the hit location (top-of-feature-normalized
  //          arriving flux)
  // true  -> absorbed flux at the hit location (incident flux weighted by the
  //          effective sticking probability)
  //
  // This flag does NOT enable/disable sticking during ray transport.
  // Sticking still affects the reflected ray payload in surfaceReflection().
  bool storeAbsorbedFlux = false;

  // Deprecated legacy name kept for compatibility. Historically this suggested
  // that sticking was either included or excluded from the transport, which is
  // misleading. It only affects how local fluxes are accumulated.
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

    // Chemical etching. k_sigma represents the intrinsic F-driven etch rate on
    // bare SiGe, while k_sigma_passivated is the corresponding effective rate
    // when the surface is strongly passivated / Ge-enriched.
    NumericType k_sigma =
        this->k_sigma_SiGe(x); // F chemical etching rate on SiGe (1e15 cm⁻²s⁻¹)
    NumericType k_sigma_passivated = this->k_sigma_SiGe(1.);
    NumericType beta_sigma =
        4.0e-2; // Passivation relaxation/removal rate for SiGe (1e15 cm⁻²s⁻¹)

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
    NumericType beta_sigma = 4.0e-2; // passivation relaxation/removal rate
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

    // put everything into the metadata
    metaData["ionFlux"] = {ionFlux};
    metaData["etchantFlux"] = {etchantFlux};
    metaData["oxygenFlux"] = {oxygenFlux};
    metaData["polymerFlux"] = {polymerFlux};
    metaData["storeAbsorbedFlux"] = {storeAbsorbedFlux ? 1.0 : 0.0};
    metaData["fluxIncludeSticking"] = {fluxIncludeSticking ? 1.0 : 0.0};
    metaData["MaskRho"] = {Mask.rho};
    metaData["MaskEthSp"] = {Mask.Eth_sp};
    metaData["MaskASp"] = {Mask.A_sp};
    metaData["SiGeRho"] = {SiGe.rho};
    metaData["SiGeEthSp"] = {SiGe.Eth_sp};
    metaData["SiGeEthIe"] = {SiGe.Eth_ie};
    metaData["SiGeASp"] = {SiGe.A_sp};
    metaData["SiGeAIe"] = {SiGe.A_ie};
    metaData["SiGeKSigma"] = {SiGe.k_sigma};
    metaData["SiGeKPassivatedSigma"] = {SiGe.k_sigma_passivated};
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
      metaData["gamma_F_" + MaterialMap::toString(gamma.material)] = {
          gamma.value};
    }
    for (const auto &gamma : gamma_F_oxidized) {
      metaData["gamma_F_oxidized_" + MaterialMap::toString(gamma.material)] = {
          gamma.value};
    }
    for (const auto &gamma : gamma_O) {
      metaData["gamma_O_" + MaterialMap::toString(gamma.material)] = {
          gamma.value};
    }
    for (const auto &gamma : gamma_O_passivated) {
      metaData["gamma_O_passivated_" + MaterialMap::toString(gamma.material)] =
          {gamma.value};
    }
    for (const auto &gamma : gamma_C) {
      metaData["gamma_C_" + MaterialMap::toString(gamma.material)] = {
          gamma.value};
    }
    for (const auto &gamma : gamma_C_oxidized) {
      metaData["gamma_C_oxidized_" + MaterialMap::toString(gamma.material)] = {
          gamma.value};
    }

    return metaData;
  }
};

#ifdef VIENNACORE_COMPILE_GPU
struct CF4O2ParametersGPU {
  CF4O2ParametersGPU() = default;

  template <typename NumericType>
  explicit CF4O2ParametersGPU(const CF4O2Parameters<NumericType> &parameters) {
    storeAbsorbedFlux = parameters.storeAbsorbedFlux ||
                        parameters.fluxIncludeSticking;

    fillMaterialMap(gamma_F, parameters.gamma_F, parameters.gamma_F.getDefault());
    fillMaterialMap(gamma_F_oxidized, parameters.gamma_F_oxidized,
                    parameters.gamma_F_oxidized.getDefault());
    fillMaterialMap(gamma_O, parameters.gamma_O, parameters.gamma_O.getDefault());
    fillMaterialMap(gamma_O_passivated, parameters.gamma_O_passivated,
                    parameters.gamma_O_passivated.getDefault());
    fillMaterialMap(gamma_C, parameters.gamma_C, parameters.gamma_C.getDefault());
    fillMaterialMap(gamma_C_oxidized, parameters.gamma_C_oxidized,
                    parameters.gamma_C_oxidized.getDefault());

    Mask.rho = static_cast<float>(parameters.Mask.rho);
    Mask.Eth_sp = static_cast<float>(parameters.Mask.Eth_sp);
    Mask.A_sp = static_cast<float>(parameters.Mask.A_sp);

    SiGe.x = static_cast<float>(parameters.SiGe.x);
    SiGe.rho = static_cast<float>(parameters.SiGe.rho);
    SiGe.Eth_sp = static_cast<float>(parameters.SiGe.Eth_sp);
    SiGe.Eth_ie = static_cast<float>(parameters.SiGe.Eth_ie);
    SiGe.A_sp = static_cast<float>(parameters.SiGe.A_sp);
    SiGe.A_ie = static_cast<float>(parameters.SiGe.A_ie);
    SiGe.k_sigma = static_cast<float>(parameters.SiGe.k_sigma);
    SiGe.k_sigma_passivated =
        static_cast<float>(parameters.SiGe.k_sigma_passivated);
    SiGe.beta_sigma = static_cast<float>(parameters.SiGe.beta_sigma);

    Si.rho = static_cast<float>(parameters.Si.rho);
    Si.Eth_sp = static_cast<float>(parameters.Si.Eth_sp);
    Si.Eth_ie = static_cast<float>(parameters.Si.Eth_ie);
    Si.A_sp = static_cast<float>(parameters.Si.A_sp);
    Si.A_ie = static_cast<float>(parameters.Si.A_ie);
    Si.k_sigma = static_cast<float>(parameters.Si.k_sigma);
    Si.beta_sigma = static_cast<float>(parameters.Si.beta_sigma);

    Passivation.Eth_O_ie =
        static_cast<float>(parameters.Passivation.Eth_O_ie);
    Passivation.Eth_C_ie =
        static_cast<float>(parameters.Passivation.Eth_C_ie);
    Passivation.A_O_ie = static_cast<float>(parameters.Passivation.A_O_ie);
    Passivation.A_C_ie = static_cast<float>(parameters.Passivation.A_C_ie);

    Ions.meanEnergy = static_cast<float>(parameters.Ions.meanEnergy);
    Ions.sigmaEnergy = static_cast<float>(parameters.Ions.sigmaEnergy);
    Ions.exponent = static_cast<float>(parameters.Ions.exponent);
    Ions.inflectAngle = static_cast<float>(parameters.Ions.inflectAngle);
    Ions.n_l = static_cast<float>(parameters.Ions.n_l);
    Ions.minAngle = static_cast<float>(parameters.Ions.minAngle);
  }

  bool storeAbsorbedFlux = false;

  float gamma_F[kBuiltInMaterialMaxId + 1]{};
  float gamma_F_oxidized[kBuiltInMaterialMaxId + 1]{};
  float gamma_O[kBuiltInMaterialMaxId + 1]{};
  float gamma_O_passivated[kBuiltInMaterialMaxId + 1]{};
  float gamma_C[kBuiltInMaterialMaxId + 1]{};
  float gamma_C_oxidized[kBuiltInMaterialMaxId + 1]{};

  CF4O2Parameters<float>::MaskType Mask;
  CF4O2Parameters<float>::SiGeType SiGe;
  CF4O2Parameters<float>::SiType Si;
  CF4O2Parameters<float>::PassivationType Passivation;
  CF4O2Parameters<float>::IonType Ions;

private:
  template <typename NumericType>
  static void fillMaterialMap(float (&out)[kBuiltInMaterialMaxId + 1],
                              const MaterialValueMap<NumericType> &source,
                              const NumericType defaultValue) {
    for (std::size_t i = 0; i <= kBuiltInMaterialMaxId; ++i) {
      out[i] = static_cast<float>(defaultValue);
    }
    for (const auto &entry : source) {
      if (entry.material.isBuiltIn()) {
        out[static_cast<std::size_t>(entry.material.builtIn())] =
            static_cast<float>(entry.value);
      }
    }
  }
};
#endif

} // namespace viennaps
