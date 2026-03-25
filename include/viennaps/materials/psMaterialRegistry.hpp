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

struct CustomMaterialInfo {
  std::string name;
};

class MaterialRegistry {
public:
  MaterialRegistry() = default;

  MaterialRegistry(const MaterialRegistry &other) {
    std::scoped_lock lock(other.mutex_);
    nameToCustomId_ = other.nameToCustomId_;
    materials_ = other.materials_;
  }

  MaterialRegistry &operator=(const MaterialRegistry &other) {
    if (this == &other) {
      return *this;
    }

    std::scoped_lock lock(mutex_, other.mutex_);
    nameToCustomId_ = other.nameToCustomId_;
    materials_ = other.materials_;
    return *this;
  }

  [[nodiscard]] Material registerMaterial(std::string name) {
    std::lock_guard<std::mutex> lock(mutex_);

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

    const auto customId = static_cast<Material::ValueType>(materials_.size());
    nameToCustomId_.emplace(name, customId);
    materials_.push_back(CustomMaterialInfo{std::move(name)});
    return Material::custom(customId);
  }

  [[nodiscard]] bool hasMaterial(std::string_view name) const {
    std::lock_guard<std::mutex> lock(mutex_);

    BuiltInMaterial builtIn = BuiltInMaterial::Undefined;
    if (tryBuiltInMaterialFromString(name, builtIn)) {
      return true;
    }

    return nameToCustomId_.find(std::string(name)) != nameToCustomId_.end();
  }

  [[nodiscard]] std::optional<Material>
  findMaterial(std::string_view name) const {
    std::lock_guard<std::mutex> lock(mutex_);

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

    std::lock_guard<std::mutex> lock(mutex_);
    const auto customId = material.customId();
    if (customId >= materials_.size()) {
      throw std::out_of_range("Unknown custom material id.");
    }

    return materials_[customId].name;
  }

  [[nodiscard]] bool isBuiltIn(Material material) const {
    return material.isBuiltIn();
  }

  [[nodiscard]] const CustomMaterialInfo &getInfo(Material material) const {
    if (material.isBuiltIn()) {
      throw std::invalid_argument(
          "Built-in materials do not have registry-owned metadata.");
    }

    std::lock_guard<std::mutex> lock(mutex_);
    const auto customId = material.customId();
    if (customId >= materials_.size()) {
      throw std::out_of_range("Unknown custom material id.");
    }

    return materials_[customId];
  }

  [[nodiscard]] std::size_t customMaterialCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return materials_.size();
  }

private:
  mutable std::mutex mutex_;
  std::unordered_map<std::string, Material::ValueType> nameToCustomId_;
  std::vector<CustomMaterialInfo> materials_;
};

} // namespace viennaps
