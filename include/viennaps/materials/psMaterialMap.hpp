#pragma once

#ifndef __CUDACC__
#include <lsMaterialMap.hpp>
#endif

#include "psBuiltInMaterial.hpp"
#include "psMaterial.hpp"
#include "psMaterialRegistry.hpp"

#include <vcLogger.hpp>
#include <vcSmartPointer.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>

namespace viennaps {

using namespace viennacore;

[[nodiscard]] constexpr MaterialCategory categoryOf(const Material material) {
  return material.isBuiltIn() ? categoryOf(material.builtIn())
                              : MaterialCategory::Generic;
}

[[nodiscard]] constexpr double density(const Material material) {
  return material.isBuiltIn() ? density(material.builtIn()) : 0.0;
}

[[nodiscard]] constexpr bool isConductive(const Material material) {
  return material.isBuiltIn() ? isConductive(material.builtIn()) : false;
}

[[nodiscard]] constexpr uint32_t color(const Material material) {
  return material.isBuiltIn() ? color(material.builtIn()) : 0xffffff;
}

/// A class that wraps the viennals MaterialMap class and provides a more user
/// friendly interface. It also provides a mapping from the integer material id
/// to the Material enum.
class MaterialMap {
#ifndef __CUDACC__
  SmartPointer<viennals::MaterialMap> map_;

public:
  MaterialMap() { map_ = SmartPointer<viennals::MaterialMap>::New(); };

  void insertNextMaterial(Material material) {
    map_->insertNextMaterial(static_cast<int>(material));
  }

  void removeMaterial() { map_->removeLastMaterial(); }

  // Returns the material **ID** at the given index. If the index is out of
  // bounds, it returns -1
  [[nodiscard]] int getMaterialIdAtIdx(std::size_t idx) const {
    return map_->getMaterialId(idx);
  }

  // Returns the material **enum** at the given index. If the index is out of
  // bounds, it returns Material::Undefined
  [[nodiscard]] Material getMaterialAtIdx(std::size_t idx) const {
    if (int id = getMaterialIdAtIdx(idx); id < 0) {
      VIENNACORE_LOG_WARNING("Getting material with out-of-bounds index.");
      return Material::Undefined;
    } else {
      return mapToMaterial(id);
    }
  }

  void setMaterialAtIdx(std::size_t idx, const Material material) {
    if (idx >= size()) {
      VIENNACORE_LOG_ERROR("Setting material with out-of-bounds index.");
      return;
    }
    map_->setMaterialId(idx, static_cast<int>(material));
  }

  [[nodiscard]] SmartPointer<viennals::MaterialMap> getMaterialMap() const {
    return map_;
  }

  [[nodiscard]] inline std::size_t size() const {
    return map_->getNumberOfLayers();
  }
#endif

public:
  __both__ static inline bool isValidMaterial(const Material mat) {
    if (mat.isBuiltIn()) {
      return isValidBuiltInMaterial(mat.builtIn());
    }
    return true;
  }

  __both__ static inline Material mapToMaterial(const int matId) {
    auto mat = Material::fromLegacyId(matId);
    return mat;
  }

  template <class T>
  __both__ static inline Material mapToMaterial(const T matId) {
    return mapToMaterial(static_cast<int>(matId));
  }

  template <class T>
  __both__ static inline bool isMaterial(const T matId,
                                         const Material material) {
    return mapToMaterial(matId) == material;
  }

  template <class T>
  __both__ static inline bool isMaterial(T matId,
                                         std::span<const Material> materials) {
    const auto material = mapToMaterial(matId);
    return std::any_of(materials.begin(), materials.end(),
                       [&](const Material &m) { return material == m; });
  }

  template <class T>
  __both__ static inline bool
  isMaterial(T matId, std::initializer_list<Material> materials) {
    return isMaterial(
        matId, std::span<const Material>(materials.begin(), materials.size()));
  }

  static inline bool isHardmask(const Material material) {
    return categoryOf(material) == MaterialCategory::Hardmask;
  }

  template <class T> static inline bool isHardmask(const T matId) {
    const auto material = mapToMaterial(matId);
    return isHardmask(material);
  }

  static inline std::string toString(const Material material) {
    if (material.isBuiltIn()) {
      return std::string(builtInMaterialToString(material.builtIn()));
    }
    auto &registry = MaterialRegistry::instance();
    if (registry.hasMaterial(material)) {
      return std::string(registry.getName(material));
    }
    return "Custom#" + std::to_string(material.customId());
  }

  static inline std::string toString(const int matId) {
    return toString(mapToMaterial(matId));
  }

  static inline Material fromString(std::string_view name) {
    BuiltInMaterial builtIn = BuiltInMaterial::Undefined;
    if (tryBuiltInMaterialFromString(name, builtIn)) {
      return Material(builtIn);
    }
    return MaterialRegistry::instance().registerMaterial(std::string(name));
  }
};

} // namespace viennaps
