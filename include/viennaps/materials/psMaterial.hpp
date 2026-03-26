#pragma once

#include "psBuiltInMaterial.hpp"
#include <vcUtil.hpp>

#include <cassert>
#include <compare>
#include <cstdint>
#include <functional>
#include <stdexcept>

namespace viennaps {

class Material {
public:
  using ValueType = std::uint32_t;

  enum class Kind : std::uint8_t { BuiltIn = 0, Custom = 1 };

  __both__ constexpr Material() noexcept = default;

  __both__ constexpr Material(BuiltInMaterial material) noexcept
      : kind_(Kind::BuiltIn), value_(static_cast<ValueType>(material)) {}

  __both__ constexpr Material(int legacyId) noexcept {
    *this = fromLegacyId(legacyId);
  }

  [[nodiscard]] __both__ static constexpr Material
  custom(ValueType id) noexcept {
    Material material;
    material.kind_ = Kind::Custom;
    material.value_ = id;
    return material;
  }

  [[nodiscard]] __both__ static constexpr Material
  fromLegacyId(int id) noexcept {
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

  [[nodiscard]] __both__ constexpr Kind kind() const noexcept { return kind_; }

  [[nodiscard]] __both__ constexpr bool isBuiltIn() const noexcept {
    return kind_ == Kind::BuiltIn;
  }

  [[nodiscard]] __both__ constexpr bool isCustom() const noexcept {
    return kind_ == Kind::Custom;
  }

  [[nodiscard]] __both__ constexpr BuiltInMaterial builtIn() const {
    if (!isBuiltIn()) {
#ifdef __CUDA_ARCH__
      assert(false && "Material is not built-in.");
      return BuiltInMaterial::Undefined;
#else
      throw std::logic_error("Material is not built-in.");
#endif
    }
    return static_cast<BuiltInMaterial>(value_);
  }

  [[nodiscard]] __both__ constexpr ValueType customId() const {
    if (!isCustom()) {
#ifdef __CUDA_ARCH__
      assert(false && "Material is not custom.");
      return 0;
#else
      throw std::logic_error("Material is not custom.");
#endif
    }
    return value_;
  }

  [[nodiscard]] __both__ constexpr ValueType legacyId() const noexcept {
    if (isBuiltIn()) {
      return value_;
    }
    return static_cast<ValueType>(kBuiltInMaterialMaxId) + 1u + value_;
  }

  __both__ constexpr operator ValueType() const noexcept { return legacyId(); }

  __both__ explicit constexpr operator int() const noexcept {
    return static_cast<int>(legacyId());
  }

  [[nodiscard]] __both__ constexpr ValueType value() const noexcept {
    return value_;
  }

  [[nodiscard]] __both__ constexpr bool
  operator==(const Material &) const noexcept = default;

  [[nodiscard]] __both__ constexpr auto
  operator<=>(const Material &) const noexcept = default;

#define MATERIAL_CONST(id, sym, cat, dens, cond, color)                        \
  static const Material sym;
  BUILTIN_MATERIAL_LIST(MATERIAL_CONST)
#undef MATERIAL_CONST

private:
  Kind kind_{Kind::BuiltIn};
  ValueType value_{static_cast<ValueType>(BuiltInMaterial::Undefined)};
};

[[nodiscard]] __both__ inline constexpr bool
isBuiltIn(const Material material) {
  return material.isBuiltIn();
}

[[nodiscard]] __both__ inline constexpr bool isCustom(const Material material) {
  return material.isCustom();
}

[[nodiscard]] __both__ inline BuiltInMaterial
toBuiltInMaterial(const Material material) {
  return material.builtIn();
}

[[nodiscard]] __both__ inline constexpr Material
toMaterial(BuiltInMaterial material) {
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
