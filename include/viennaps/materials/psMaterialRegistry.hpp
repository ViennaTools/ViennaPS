#pragma once

#include "psMaterial.hpp"

#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace viennaps {

class MaterialRegistry {
public:
  [[nodiscard]] static MaterialRegistry &instance() {
    static MaterialRegistry registry;
    return registry;
  }

  MaterialRegistry(const MaterialRegistry &) = delete;
  MaterialRegistry &operator=(const MaterialRegistry &) = delete;
  MaterialRegistry(MaterialRegistry &&) = delete;
  MaterialRegistry &operator=(MaterialRegistry &&) = delete;

  [[nodiscard]] Material registerMaterial(std::string name) {

    if (name.empty()) {
      throw std::invalid_argument("Material name must not be empty.");
    }

    BuiltInMaterial builtIn = BuiltInMaterial::Undefined;
    if (tryBuiltInMaterialFromString(name, builtIn)) {
      return Material(builtIn);
    }

    const auto it = nameToCustomId_.find(name);
    if (it != nameToCustomId_.end()) {
      return Material::custom(it->second);
    }

    // register new custom material
    const auto customId = static_cast<Material::ValueType>(materials_.size());
    nameToCustomId_.emplace(name, customId);
    materials_.push_back(MaterialInfo{
        std::move(name), MaterialCategory::Generic, 0.0, false, 0xffffff});
    return Material::custom(customId);
  }

  [[nodiscard]] bool hasMaterial(std::string_view name) const {

    BuiltInMaterial builtIn = BuiltInMaterial::Undefined;
    if (tryBuiltInMaterialFromString(name, builtIn)) {
      return true;
    }

    return nameToCustomId_.find(std::string(name)) != nameToCustomId_.end();
  }

  [[nodiscard]] bool hasMaterial(Material material) const {
    if (material.isBuiltIn()) {
      return true;
    }
    const auto customId = material.customId();
    return customId < materials_.size();
  }

  [[nodiscard]] std::optional<Material>
  findMaterial(std::string_view name) const {

    BuiltInMaterial builtIn = BuiltInMaterial::Undefined;
    if (tryBuiltInMaterialFromString(name, builtIn)) {
      return Material(builtIn);
    }

    const auto it = nameToCustomId_.find(std::string(name));
    if (it == nameToCustomId_.end()) {
      return std::nullopt;
    }

    return Material::custom(it->second);
  }

  [[nodiscard]] Material getMaterial(std::string_view name) const {
    const auto maybe = findMaterial(name);
    if (!maybe.has_value()) {
      throw std::invalid_argument("Unknown material: " + std::string(name));
    }
    return maybe.value();
  }

  [[nodiscard]] std::string_view getName(Material material) const {
    if (material.isBuiltIn()) {
      return builtInMaterialToString(material.builtIn());
    }

    const auto customId = material.customId();
    if (customId >= materials_.size()) {
      throw std::out_of_range("Unknown custom material id.");
    }

    return materials_[customId].name;
  }

  [[nodiscard]] bool isBuiltIn(Material material) const {
    return material.isBuiltIn();
  }

  [[nodiscard]] MaterialInfo getInfo(Material material) const {
    if (material.isBuiltIn()) {
      auto info = getBuiltInMaterialInfo(material.builtIn());
      return MaterialInfo{std::string(info.name), info.category,
                          info.density_gcm3, info.conductive, info.colorHex};
    }

    const auto customId = material.customId();
    if (customId >= materials_.size()) {
      throw std::out_of_range("Unknown custom material id.");
    }

    return materials_[customId];
  }

  void setInfo(Material material, const MaterialInfo &info) {
    if (material.isBuiltIn()) {
      throw std::invalid_argument(
          "Cannot set metadata for built-in materials.");
    }

    const auto customId = material.customId();
    if (customId >= materials_.size()) {
      throw std::out_of_range("Unknown custom material id.");
    }

    materials_[customId] = info;
  }

  [[nodiscard]] std::size_t customMaterialCount() const {
    return materials_.size();
  }

private:
  MaterialRegistry() = default;

  std::unordered_map<std::string, Material::ValueType> nameToCustomId_;
  std::vector<MaterialInfo> materials_;
};

} // namespace viennaps
