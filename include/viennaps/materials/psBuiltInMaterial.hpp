#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace viennaps {

enum class MaterialCategory : uint8_t {
  Generic,
  Silicon,
  OxideNitride,
  Hardmask,
  Metal,
  Silicide,
  Compound,
  TwoD,
  TCO,
  Misc
};

template <bool IsBuiltIn> struct MaterialInfoType {
  using StringType =
      std::conditional_t<IsBuiltIn, std::string_view, std::string>;
  StringType name;
  MaterialCategory category;
  double density_gcm3;
  bool conductive;
  uint32_t colorHex;
};

using BuiltInMaterialInfo = MaterialInfoType<true>;
using MaterialInfo = MaterialInfoType<false>;

#define BUILTIN_MATERIAL_LIST(X)                                               \
  /* id, sym, cat, density_gcm3, conductive, color (hex) */                    \
  X(0, Mask, Hardmask, 500.0, false, 0x333333)                                 \
  X(1, Polymer, Generic, 1.2, false, 0xed3b13)                                 \
  X(2, Air, Generic, 0.0012, false, 0x87ceeb)                                  \
  X(3, GAS, Generic, 0.001, false, 0xb0e0e6)                                   \
  X(4, Dielectric, Generic, 2.2, false, 0xaed6f1)                              \
  X(5, Metal, Metal, 7.5, true, 0xc0c0c0)                                      \
  X(6, Undefined, Generic, 0.0, false, 0xcccccc)                               \
  /* Silicon and derivatives */                                                \
  X(10, Si, Silicon, 2.33, false, 0xc79a08)                                    \
  X(11, PolySi, Silicon, 2.33, false, 0xed8713)                                \
  X(12, aSi, Silicon, 2.2, false, 0xfee315)                                    \
  X(13, SiGe, Silicon, 4.0, false, 0xd66604)                                   \
  X(14, SiC, Silicon, 3.21, false, 0x989300)                                   \
  X(15, SiN, OxideNitride, 3.1, false, 0x00bfa5)                               \
  X(16, Si3N4, OxideNitride, 3.1, false, 0x00a8a8)                             \
  X(17, SiON, OxideNitride, 2.4, false, 0x33cccc)                              \
  X(18, SiCN, OxideNitride, 2.6, false, 0x2aa198)                              \
  X(19, SiBCN, OxideNitride, 2.3, false, 0x00897b)                             \
  X(20, SiCOH, OxideNitride, 1.9, false, 0x26c6da)                             \
  X(21, SiOCN, OxideNitride, 2.1, false, 0x00acc1)                             \
  X(22, BulkSi, Silicon, 2.33, false, 0xc79a08)                                \
  /* Oxides and nitrides */                                                    \
  X(30, SiO2, OxideNitride, 2.2, false, 0x66ccff)                              \
  X(31, Al2O3, OxideNitride, 3.95, false, 0x90caf9)                            \
  X(32, HfO2, OxideNitride, 9.7, false, 0x3f51b5)                              \
  X(33, ZrO2, OxideNitride, 5.7, false, 0x5c6bc0)                              \
  X(34, TiO2, OxideNitride, 4.2, false, 0x64b5f6)                              \
  X(35, Y2O3, OxideNitride, 5.0, false, 0x42a5f5)                              \
  X(36, La2O3, OxideNitride, 6.5, false, 0x2196f3)                             \
  X(37, AlN, OxideNitride, 3.26, false, 0x00bcd4)                              \
  X(38, Ta2O5, OxideNitride, 8.2, false, 0x1976d2)                             \
  X(39, BN, OxideNitride, 2.1, false, 0x4db6ac)                                \
  X(40, hBN, OxideNitride, 2.1, false, 0x80cbc4)                               \
  /* Carbon / hardmasks and organics */                                        \
  X(50, C, Hardmask, 2.2, false, 0x212121)                                     \
  X(51, aC, Hardmask, 1.8, false, 0x303030)                                    \
  X(52, SOC, Hardmask, 1.4, false, 0x795548)                                   \
  X(53, SOG, OxideNitride, 1.4, false, 0xa1887f)                               \
  X(54, BPSG, OxideNitride, 2.2, false, 0x8d6e63)                              \
  X(55, PSG, OxideNitride, 2.2, false, 0x6d4c41)                               \
  X(56, SiLK, Hardmask, 1.0, false, 0xffb74d)                                  \
  X(57, ARC, Hardmask, 1.1, false, 0xffa000)                                   \
  X(58, PMMA, Hardmask, 1.18, false, 0xec407a)                                 \
  X(59, PHS, Hardmask, 1.2, false, 0xffe082)                                   \
  X(60, HSQ, OxideNitride, 1.3, false, 0x81d4fa)                               \
  /* Metals */                                                                 \
  X(70, W, Metal, 19.3, true, 0x6e6e6e)                                        \
  X(71, Cu, Metal, 8.96, true, 0xb87333)                                       \
  X(72, Co, Metal, 8.9, true, 0x3d5a99)                                        \
  X(73, Ru, Metal, 12.4, true, 0x7a7a7a)                                       \
  X(74, Ni, Metal, 8.9, true, 0x9c9c9c)                                        \
  X(75, Pt, Metal, 21.4, true, 0xe5e4e2)                                       \
  X(76, Ta, Metal, 16.7, true, 0x8d909b)                                       \
  X(77, TaN, Metal, 14.3, true, 0x6b6f7b)                                      \
  X(78, Ti, Metal, 4.5, true, 0xc0c0c0)                                        \
  X(79, TiN, Metal, 5.4, true, 0xadb5bd)                                       \
  X(80, Mo, Metal, 10.3, true, 0x8f8f8f)                                       \
  X(81, Ir, Metal, 22.6, true, 0xdfe0e2)                                       \
  X(82, Rh, Metal, 12.4, true, 0xd1d1d1)                                       \
  X(83, Pd, Metal, 12.0, true, 0xc9c9c9)                                       \
  X(84, RuTa, Metal, 13.5, true, 0x7b7f8a)                                     \
  X(85, CoW, Metal, 9.5, true, 0x708090)                                       \
  X(86, NiW, Metal, 9.6, true, 0x7f8c8d)                                       \
  X(87, TiAlN, Metal, 4.8, true, 0x9b8a3e)                                     \
  X(88, Mn, Metal, 7.2, true, 0x7b7b7b)                                        \
  X(89, MnO, Metal, 5.4, false, 0x556b2f)                                      \
  X(90, MnN, Metal, 6.1, false, 0x2f4f4f)                                      \
  X(91, Au, Metal, 19.3, true, 0xd4af37)                                       \
  X(92, Cr, Metal, 7.19, true, 0xa3a3a3)                                       \
  /* Silicides */                                                              \
  X(100, WSi2, Silicide, 9.3, true, 0x607d8b)                                  \
  X(101, TiSi2, Silicide, 4.0, true, 0x546e7a)                                 \
  X(102, MoSi2, Silicide, 6.3, true, 0x78909c)                                 \
  /* Compound semiconductors */                                                \
  X(110, Ge, Compound, 5.32, false, 0x808080)                                  \
  X(111, GaN, Compound, 6.15, false, 0x00ced1)                                 \
  X(112, GaAs, Compound, 5.32, false, 0x8a2be2)                                \
  X(113, InP, Compound, 4.81, false, 0xff6347)                                 \
  X(114, InGaAs, Compound, 5.6, false, 0x7b1fa2)                               \
  X(115, SiGaN, Compound, 5.0, false, 0x26a69a)                                \
  X(116, SiOCH, OxideNitride, 1.8, false, 0x4fc3f7)                            \
  /* 2D + emerging */                                                          \
  X(130, Graphene, TwoD, 2.2, true, 0x000000)                                  \
  X(131, MoS2, TwoD, 5.0, false, 0x66bb6a)                                     \
  X(132, WS2, TwoD, 7.5, false, 0xffee58)                                      \
  X(133, WSe2, TwoD, 9.3, false, 0xec407a)                                     \
  X(134, VO2, TwoD, 4.6, false, 0x8d6e63)                                      \
  X(135, GST, TwoD, 6.1, false, 0x800020)                                      \
  /* Transparent conductors */                                                 \
  X(150, ITO, TCO, 7.1, true, 0x00ffff)                                        \
  X(151, ZnO, TCO, 5.6, true, 0x98fb98)                                        \
  X(152, AZO, TCO, 5.5, true, 0x20b2aa)                                        \
  /* Misc hardmask aliases */                                                  \
  X(170, SiON_HM, Hardmask, 2.4, false, 0x33cccc)                              \
  X(171, SiN_HM, Hardmask, 3.1, false, 0x00a8a8)                               \
  X(172, SiC_HM, Hardmask, 3.2, false, 0x4f4f4f)                               \
  X(173, TiO, Misc, 4.9, false, 0x64b5f6)                                      \
  X(174, ZrO, Misc, 5.2, false, 0x5c6bc0)                                      \
  X(175, SiO2_HM, Hardmask, 2.2, false, 0x66ccff)                              \
  X(176, Custom, Generic, 0.0, false, 0xffffff)

#ifndef MATERIAL_LIST
#define MATERIAL_LIST(X) BUILTIN_MATERIAL_LIST(X)
#endif

enum class BuiltInMaterial : uint16_t {
#define ENUM_ELEM(id, sym, cat, dens, cond, color) sym = id,
  BUILTIN_MATERIAL_LIST(ENUM_ELEM)
#undef ENUM_ELEM
};

inline constexpr uint16_t kBuiltInMaterialMaxId = 176;

constexpr std::array<BuiltInMaterialInfo, kBuiltInMaterialMaxId + 1>
    kBuiltInMaterialTable = [] {
      std::array<BuiltInMaterialInfo, kBuiltInMaterialMaxId + 1> table{};
      for (auto &entry : table) {
        entry = {"Undefined", MaterialCategory::Generic, 0.0, false, 0xcccccc};
      }
#define FILL_ROW(id, sym, cat, dens, cond, color)                              \
  table[id] = {#sym, MaterialCategory::cat, dens, cond, color};
      BUILTIN_MATERIAL_LIST(FILL_ROW)
#undef FILL_ROW
      return table;
    }();

[[nodiscard]] constexpr bool isValidBuiltInMaterialId(uint16_t id) {
  return id <= kBuiltInMaterialMaxId;
}

[[nodiscard]] constexpr bool isValidBuiltInMaterial(BuiltInMaterial material) {
  return isValidBuiltInMaterialId(static_cast<uint16_t>(material));
}

[[nodiscard]] constexpr const BuiltInMaterialInfo &
getBuiltInMaterialInfo(BuiltInMaterial material) {
  const auto id = static_cast<uint16_t>(material);
  return kBuiltInMaterialTable[isValidBuiltInMaterialId(id)
                                   ? id
                                   : static_cast<uint16_t>(
                                         BuiltInMaterial::Undefined)];
}

[[nodiscard]] constexpr std::string_view
builtInMaterialToString(BuiltInMaterial material) {
  return getBuiltInMaterialInfo(material).name;
}

[[nodiscard]] constexpr MaterialCategory categoryOf(BuiltInMaterial material) {
  return getBuiltInMaterialInfo(material).category;
}

[[nodiscard]] constexpr double density(BuiltInMaterial material) {
  return getBuiltInMaterialInfo(material).density_gcm3;
}

[[nodiscard]] constexpr bool isConductive(BuiltInMaterial material) {
  return getBuiltInMaterialInfo(material).conductive;
}

[[nodiscard]] constexpr uint32_t color(BuiltInMaterial material) {
  return getBuiltInMaterialInfo(material).colorHex;
}

[[nodiscard]] inline bool tryBuiltInMaterialFromString(std::string_view name,
                                                       BuiltInMaterial &out) {
#define MATCH_NAME(id, sym, cat, dens, cond, color)                            \
  if (name == std::string_view(#sym)) {                                        \
    out = BuiltInMaterial::sym;                                                \
    return true;                                                               \
  }
  BUILTIN_MATERIAL_LIST(MATCH_NAME)
#undef MATCH_NAME
  return false;
}

[[nodiscard]] inline BuiltInMaterial
builtInMaterialFromString(std::string_view name) {
  BuiltInMaterial material = BuiltInMaterial::Undefined;
  if (!tryBuiltInMaterialFromString(name, material)) {
    throw std::invalid_argument("Unknown built-in material name: " +
                                std::string(name));
  }
  return material;
}

} // namespace viennaps
