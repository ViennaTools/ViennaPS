#pragma once

#include "../materials/psMaterialMap.hpp"

#include <vcLogger.hpp>
#include <vcRNG.hpp>

#include <cassert>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace viennaps {

template <typename NumericType> struct FluorocarbonParameters {

  struct MaterialParameters {

    Material id = Material::Undefined;

    // density
    NumericType density = 2.2; // 1e22 atoms/cm³

    // sticking
    NumericType beta_p = 0.26;
    NumericType beta_e = 0.9;

    // sputtering coefficients
    NumericType Eth_sp = 18.; // eV
    NumericType Eth_ie = 4.;  // eV
    NumericType A_sp = 0.0139;
    NumericType B_sp = 9.3;
    NumericType A_ie = 0.0361;

    // chemical etching
    NumericType K = 0.002789491704544977;
    NumericType E_a = 0.168; // eV
  };

  std::vector<MaterialParameters> materials;

  // fluxes in (1e15 /cm² /s)
  NumericType ionFlux = 56.;
  NumericType etchantFlux = 500.;
  NumericType polyFlux = 100.;

  NumericType delta_p = 1.;
  NumericType etchStopDepth = std::numeric_limits<NumericType>::lowest();

  NumericType temperature = 300.; // K
  NumericType k_ie = 2.;
  NumericType k_ev = 2.;

  struct IonType {
    NumericType meanEnergy = 100.; // eV
    NumericType sigmaEnergy = 10.; // eV
    NumericType exponent = 500.;

    NumericType inflectAngle = 1.55334303;
    NumericType n_l = 10.;
    NumericType minAngle = 1.3962634;
  } Ions;

  void addMaterial(const MaterialParameters &material) {
    materials.push_back(material);
  }

  MaterialParameters getMaterialParameters(const Material material) const {
    for (const auto &m : materials) {
      if (m.id == material)
        return m;
    }
    VIENNACORE_LOG_ERROR("Material '" + MaterialMap::toString(material) +
                         "' not found in fluorocarbon model parameters.");
    return MaterialParameters{};
  }

  auto toProcessMetaData() const {
    std::unordered_map<std::string, std::vector<double>> processData;

    processData["ionFlux"] = std::vector<double>{ionFlux};
    processData["etchantFlux"] = std::vector<double>{etchantFlux};
    processData["polymerFlux"] = std::vector<double>{polyFlux};
    processData["delta_p"] = std::vector<double>{delta_p};
    processData["etchStopDepth"] = std::vector<double>{etchStopDepth};
    processData["temperature"] = std::vector<double>{temperature};
    processData["k_ie"] = std::vector<double>{k_ie};
    processData["k_ev"] = std::vector<double>{k_ev};
    processData["Ion MeanEnergy"] = std::vector<double>{Ions.meanEnergy};
    processData["Ion SigmaEnergy"] = std::vector<double>{Ions.sigmaEnergy};
    processData["Ion Exponent"] = std::vector<double>{Ions.exponent};
    processData["Ion InflectAngle"] = std::vector<double>{Ions.inflectAngle};
    processData["Ion n_k"] = std::vector<double>{Ions.n_l};
    processData["Ion MinAngle"] = std::vector<double>{Ions.minAngle};
    for (auto mat : materials) {
      std::string prefix = MaterialMap::toString(mat.id) + " ";
      processData[prefix + "density"] = std::vector<double>{mat.density};
      processData[prefix + "beta_p"] = std::vector<double>{mat.beta_p};
      processData[prefix + "beta_e"] = std::vector<double>{mat.beta_e};
      processData[prefix + "Eth_sp"] = std::vector<double>{mat.Eth_sp};
      processData[prefix + "Eth_ie"] = std::vector<double>{mat.Eth_ie};
      processData[prefix + "A_sp"] = std::vector<double>{mat.A_sp};
      processData[prefix + "B_sp"] = std::vector<double>{mat.B_sp};
      processData[prefix + "A_ie"] = std::vector<double>{mat.A_ie};
      processData[prefix + "K"] = std::vector<double>{mat.K};
      processData[prefix + "E_a"] = std::vector<double>{mat.E_a};
    }

    return processData;
  }
};

// #ifdef VIENNACORE_COMPILE_GPU
namespace gpu {

struct FluorocarbonParameters {
  static constexpr std::uint32_t maxMaterials = 10;

  struct MaterialParameters {
    int id = static_cast<int>(Material::Undefined);

    // sticking
    float beta_p = 0.26f;
    float beta_e = 0.9f;

    // sputtering coefficients
    float Eth_sp = 18.f; // eV
    float Eth_ie = 4.f;  // eV
    float B_sp = 9.3f;

    MaterialParameters() = default;

    template <typename Parameters>
    explicit MaterialParameters(const Parameters &parameters)
        : id(static_cast<int>(parameters.id)),
          beta_p(static_cast<float>(parameters.beta_p)),
          beta_e(static_cast<float>(parameters.beta_e)),
          Eth_sp(static_cast<float>(parameters.Eth_sp)),
          Eth_ie(static_cast<float>(parameters.Eth_ie)),
          B_sp(static_cast<float>(parameters.B_sp)) {}
  };

  MaterialParameters materials[maxMaterials]{};
  std::uint32_t numMaterials = 0;

  struct IonType {
    float meanEnergy = 100.f; // eV
    float sigmaEnergy = 10.f; // eV
    float exponent = 500.f;

    float inflectAngle = 1.55334303f;
    float n_l = 10.f;
    float minAngle = 1.3962634f;
  } Ions;

  FluorocarbonParameters() = default;

#ifndef __CUDACC__
  template <typename Parameters>
  explicit FluorocarbonParameters(const Parameters &parameters) {
    set(parameters);
  }

  template <typename Parameters> void set(const Parameters &parameters) {
    Ions.meanEnergy = static_cast<float>(parameters.Ions.meanEnergy);
    Ions.sigmaEnergy = static_cast<float>(parameters.Ions.sigmaEnergy);
    Ions.exponent = static_cast<float>(parameters.Ions.exponent);
    Ions.inflectAngle = static_cast<float>(parameters.Ions.inflectAngle);
    Ions.n_l = static_cast<float>(parameters.Ions.n_l);
    Ions.minAngle = static_cast<float>(parameters.Ions.minAngle);

    auto materials = parameters.materials;
    std::sort(materials.begin(), materials.end(),
              [](const auto &a, const auto &b) { return a.id < b.id; });

    numMaterials = 0;
    for (const auto &material : materials) {
      addMaterial(MaterialParameters(material));
    }
  }

  bool addMaterial(const MaterialParameters &material) {
    if (numMaterials >= maxMaterials) {
      VIENNACORE_LOG_ERROR(
          "Fluorocarbon GPU parameters support at most 10 materials.");
      return false;
    }

    materials[numMaterials++] = material;
    return true;
  }
#endif
};

static_assert(
    std::is_trivially_copyable_v<FluorocarbonParameters::MaterialParameters>);
static_assert(std::is_trivially_copyable_v<FluorocarbonParameters>);
} // namespace gpu
// #endif

} // namespace viennaps
