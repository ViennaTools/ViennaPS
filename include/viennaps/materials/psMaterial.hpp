#pragma once

#include "psBuiltInMaterial.hpp"

#include <compare>
#include <cstdint>
#include <functional>
#include <stdexcept>

namespace viennaps {

class Material {
public:
  using ValueType = std::uint32_t;

  enum class Kind : std::uint8_t { BuiltIn = 0, Custom = 1 };

  constexpr Material() noexcept = default;

  constexpr Material(BuiltInMaterial material) noexcept
      : kind_(Kind::BuiltIn), value_(static_cast<ValueType>(material)) {}

  constexpr Material(int legacyId) noexcept { *this = fromLegacyId(legacyId); }

  [[nodiscard]] static constexpr Material custom(ValueType id) noexcept {
    Material material;
    material.kind_ = Kind::Custom;
    material.value_ = id;
    return material;
  }

  [[nodiscard]] static constexpr Material fromLegacyId(int id) noexcept {
    if (id < 0) {
      return Material(BuiltInMaterial::Undefined);
    }

    const auto raw = static_cast<ValueType>(id);
    if (raw <= static_cast<ValueType>(kBuiltInMaterialMaxId)) {
      return Material(static_cast<BuiltInMaterial>(raw));
    }

    return Material::custom(raw -
                            static_cast<ValueType>(kBuiltInMaterialMaxId) - 1u);
  }

  [[nodiscard]] constexpr Kind kind() const noexcept { return kind_; }

  [[nodiscard]] constexpr bool isBuiltIn() const noexcept {
    return kind_ == Kind::BuiltIn;
  }

  [[nodiscard]] constexpr bool isCustom() const noexcept {
    return kind_ == Kind::Custom;
  }

  [[nodiscard]] constexpr BuiltInMaterial builtIn() const {
    if (!isBuiltIn()) {
      throw std::logic_error("Material is not built-in.");
    }
    return static_cast<BuiltInMaterial>(value_);
  }

  [[nodiscard]] constexpr ValueType customId() const {
    if (!isCustom()) {
      throw std::logic_error("Material is not custom.");
    }
    return value_;
  }

  [[nodiscard]] constexpr ValueType legacyId() const noexcept {
    if (isBuiltIn()) {
      return value_;
    }
    return static_cast<ValueType>(kBuiltInMaterialMaxId) + 1u + value_;
  }

  constexpr operator ValueType() const noexcept { return legacyId(); }

  explicit constexpr operator int() const noexcept {
    return static_cast<int>(legacyId());
  }

  [[nodiscard]] constexpr ValueType value() const noexcept { return value_; }

  [[nodiscard]] constexpr bool
  operator==(const Material &) const noexcept = default;

  [[nodiscard]] constexpr auto
  operator<=>(const Material &) const noexcept = default;

#define MATERIAL_CONST(id, sym, cat, dens, cond, color)                        \
  static const Material sym;
  BUILTIN_MATERIAL_LIST(MATERIAL_CONST)
#undef MATERIAL_CONST

private:
  Kind kind_{Kind::BuiltIn};
  ValueType value_{static_cast<ValueType>(BuiltInMaterial::Undefined)};
};

[[nodiscard]] inline constexpr bool isBuiltIn(const Material material) {
  return material.isBuiltIn();
}

[[nodiscard]] inline constexpr bool isCustom(const Material material) {
  return material.isCustom();
}

[[nodiscard]] inline BuiltInMaterial
toBuiltInMaterial(const Material material) {
  return material.builtIn();
}

[[nodiscard]] inline constexpr Material toMaterial(BuiltInMaterial material) {
  return Material(material);
}

} // namespace viennaps

namespace viennaps {

#define MATERIAL_CONST_DEF(id, sym, cat, dens, cond, color)                    \
  inline constexpr Material Material::sym{BuiltInMaterial::sym};
BUILTIN_MATERIAL_LIST(MATERIAL_CONST_DEF)
#undef MATERIAL_CONST_DEF

} // namespace viennaps

namespace std {

template <> struct hash<viennaps::Material> {
  size_t operator()(const viennaps::Material &material) const noexcept {
    const auto mix = (static_cast<std::uint64_t>(material.value()) << 8u) |
                     static_cast<std::uint8_t>(material.kind());
    return std::hash<std::uint64_t>{}(mix);
  }
};

} // namespace std
