#pragma once

#ifdef VIENNACORE_COMPILE_GPU
#include "../materials/psMaterial.hpp"

#include <vcLogger.hpp>
#include <vcRNG.hpp>

#include <cassert>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace viennaps::gpu {

struct FluorocarbonParameters {
  static constexpr std::uint32_t maxMaterials = 5;

  struct MaterialParameters {
    // sticking
    float beta_p = 0.26f;
    float beta_e = 0.9f;

    // sputtering coefficients
    float Eth_sp = 18.f; // eV
    float Eth_ie = 4.f;  // eV
    float A_sp = 0.0139f;
    float B_sp = 9.3f;
    float A_ie = 0.0361f;

    MaterialParameters() = default;

    template <typename Parameters>
    explicit MaterialParameters(const Parameters &parameters)
        : beta_p(static_cast<float>(parameters.beta_p)),
          beta_e(static_cast<float>(parameters.beta_e)),
          Eth_sp(static_cast<float>(parameters.Eth_sp)),
          Eth_ie(static_cast<float>(parameters.Eth_ie)),
          A_sp(static_cast<float>(parameters.A_sp)),
          B_sp(static_cast<float>(parameters.B_sp)),
          A_ie(static_cast<float>(parameters.A_ie)) {}
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

    numMaterials = 0;
    for (const auto &material : parameters.materials) {
      addMaterial(MaterialParameters(material));
    }
  }

  __both__ bool addMaterial(const MaterialParameters &material) {
    if (numMaterials >= maxMaterials) {
#ifdef __CUDA_ARCH__
      assert(false && "Too many fluorocarbon materials for GPU parameters.");
#else
      VIENNACORE_LOG_ERROR(
          "Fluorocarbon GPU parameters support at most 5 materials.");
#endif
      return false;
    }

    materials[numMaterials++] = material;
    return true;
  }

  //   __both__ MaterialParameters
  //   getMaterialParameters(const Material material) const {
  //     for (std::uint32_t i = 0; i < numMaterials; ++i) {
  //       if (materials[i].id == material)
  //         return materials[i];
  //     }

  // #ifdef __CUDA_ARCH__
  //     assert(false && "Material not found in fluorocarbon GPU parameters.");
  // #else
  //     VIENNACORE_LOG_ERROR("Material not found in fluorocarbon GPU
  //     parameters.");
  // #endif
  //     return MaterialParameters{};
  //   }

  //   __both__ MaterialParameters
  //   getMaterialParameters(const int materialId) const {
  //     return getMaterialParameters(Material::fromLegacyId(materialId));
  //   }
};

static_assert(
    std::is_trivially_copyable_v<FluorocarbonParameters::MaterialParameters>);
static_assert(std::is_trivially_copyable_v<FluorocarbonParameters>);

} // namespace viennaps::gpu
#endif
