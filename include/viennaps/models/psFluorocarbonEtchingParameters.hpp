#pragma once

#include <psMaterials.hpp>
#include <vcLogger.hpp>

namespace viennaps {

using namespace viennacore;

// Parameters from:
// A. LaMagna and G. Garozzo "Factors affecting profile evolution in plasma
// etching of SiO2: Modeling and experimental verification" Journal of the
// Electrochemical Society 150(10) 2003 pp. 1896-1902

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
    Logger::getInstance()
        .addError("Material '" + MaterialMap::toString(material) +
                  "' not found in fluorocarbon model parameters.")
        .print();
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

namespace gpu {
struct FluorocarbonEtchingParametersStaticF {

  float temperature;
  float k_ie;
  float k_ev;

  float ionMeanEnergy;
  float ionSigmaEnergy;
  float ionExponent;
  float ionInflectAngle;
  float ionN_l;
  float ionMinAngle;
};
} // namespace gpu

} // namespace viennaps