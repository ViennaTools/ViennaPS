#include "models/psFluorocarbonParameters.hpp"

using namespace viennaps;

int main() {
  using NumericType = double;

  // use pre-defined Fluorocarbon etching model
  auto parameters = FluorocarbonParameters<NumericType>();
  parameters.addMaterial({.id = Material::Si, .density = 5.5});
  parameters.addMaterial({.id = Material::SiO2, .density = 2.2});
  parameters.addMaterial({.id = Material::Si3N4, .density = 2.3});
  parameters.addMaterial({.id = Material::Polymer,
                          .density = 2.,
                          .beta_e = 0.6,
                          .A_ie = 0.0361 * 2});
  parameters.addMaterial({.id = Material::Mask,
                          .density = 500.,
                          .beta_p = 0.01,
                          .beta_e = 0.1,
                          .Eth_sp = 20.});

  parameters.ionFlux = 10.0;
  parameters.etchantFlux = 5.0;
  parameters.polyFlux = 2.0;
  parameters.Ions.meanEnergy = 100.0;
  parameters.Ions.sigmaEnergy = 10.0;
  parameters.Ions.exponent = 500.0;

  auto gpuData = gpu::FluorocarbonParameters(parameters);

  for (size_t i = 0; i < gpuData.numMaterials; ++i) {
    auto mat = gpuData.materials[i];
    std::cout << "Material ID: " << mat.id << std::endl;
  }
}