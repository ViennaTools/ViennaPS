#pragma once

#ifndef __CUDACC__
#include <lsMaterialMap.hpp>
#endif

#include <vcLogger.hpp>
#include <vcSmartPointer.hpp>

#include <array>
#include <string>
#include <string_view>

namespace viennaps {

using namespace viennacore;

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

#define MATERIAL_LIST(X)                                                       \
  /* id, sym, cat, density_gcm3, conductive */                                 \
  X(0, Mask, Hardmask, 500.0, false)                                           \
  X(1, Polymer, Generic, 1.2, false)                                           \
  X(2, Air, Generic, 0.0012, false)                                            \
  X(3, GAS, Generic, 0.001, false)                                             \
  X(4, Dielectric, Generic, 2.2, false)                                        \
  X(5, Metal, Metal, 7.5, true)                                                \
  X(6, Undefined, Generic, 0.0, false)                                         \
  /* Silicon and derivatives */                                                \
  X(10, Si, Silicon, 2.33, false)                                              \
  X(11, PolySi, Silicon, 2.33, false)                                          \
  X(12, aSi, Silicon, 2.2, false)                                              \
  X(13, SiGe, Silicon, 4.0, false)                                             \
  X(14, SiC, Silicon, 3.21, false)                                             \
  X(15, SiN, OxideNitride, 3.1, false)                                         \
  X(16, Si3N4, OxideNitride, 3.1, false)                                       \
  X(17, SiON, OxideNitride, 2.4, false)                                        \
  X(18, SiCN, OxideNitride, 2.6, false)                                        \
  X(19, SiBCN, OxideNitride, 2.3, false)                                       \
  X(20, SiCOH, OxideNitride, 1.9, false)                                       \
  X(21, SiOCN, OxideNitride, 2.1, false)                                       \
  /* Oxides and nitrides */                                                    \
  X(30, SiO2, OxideNitride, 2.2, false)                                        \
  X(31, Al2O3, OxideNitride, 3.95, false)                                      \
  X(32, HfO2, OxideNitride, 9.7, false)                                        \
  X(33, ZrO2, OxideNitride, 5.7, false)                                        \
  X(34, TiO2, OxideNitride, 4.2, false)                                        \
  X(35, Y2O3, OxideNitride, 5.0, false)                                        \
  X(36, La2O3, OxideNitride, 6.5, false)                                       \
  X(37, AlN, OxideNitride, 3.26, false)                                        \
  X(38, Ta2O5, OxideNitride, 8.2, false)                                       \
  X(39, BN, OxideNitride, 2.1, false)                                          \
  X(40, hBN, OxideNitride, 2.1, false)                                         \
  /* Carbon / hardmasks and organics */                                        \
  X(50, C, Hardmask, 2.2, false)                                               \
  X(51, aC, Hardmask, 1.8, false)                                              \
  X(52, SOC, Hardmask, 1.4, false)                                             \
  X(53, SOG, OxideNitride, 1.4, false)                                         \
  X(54, BPSG, OxideNitride, 2.2, false)                                        \
  X(55, PSG, OxideNitride, 2.2, false)                                         \
  X(56, SiLK, Hardmask, 1.0, false)                                            \
  X(57, ARC, Hardmask, 1.1, false)                                             \
  X(58, PMMA, Hardmask, 1.18, false)                                           \
  X(59, PHS, Hardmask, 1.2, false)                                             \
  X(60, HSQ, OxideNitride, 1.3, false)                                         \
  /* Metals */                                                                 \
  X(70, W, Metal, 19.3, true)                                                  \
  X(71, Cu, Metal, 8.96, true)                                                 \
  X(72, Co, Metal, 8.9, true)                                                  \
  X(73, Ru, Metal, 12.4, true)                                                 \
  X(74, Ni, Metal, 8.9, true)                                                  \
  X(75, Pt, Metal, 21.4, true)                                                 \
  X(76, Ta, Metal, 16.7, true)                                                 \
  X(77, TaN, Metal, 14.3, true)                                                \
  X(78, Ti, Metal, 4.5, true)                                                  \
  X(79, TiN, Metal, 5.4, true)                                                 \
  X(80, Mo, Metal, 10.3, true)                                                 \
  X(81, Ir, Metal, 22.6, true)                                                 \
  X(82, Rh, Metal, 12.4, true)                                                 \
  X(83, Pd, Metal, 12.0, true)                                                 \
  X(84, RuTa, Metal, 13.5, true)                                               \
  X(85, CoW, Metal, 9.5, true)                                                 \
  X(86, NiW, Metal, 9.6, true)                                                 \
  X(87, TiAlN, Metal, 4.8, true)                                               \
  X(88, Mn, Metal, 7.2, true)                                                  \
  X(89, MnO, Metal, 5.4, false)                                                \
  X(90, MnN, Metal, 6.1, false)                                                \
  X(91, Au, Metal, 19.3, true)                                                 \
  X(92, Cr, Metal, 7.19, true)                                                 \
  /* Silicides */                                                              \
  X(100, WSi2, Silicide, 9.3, true)                                            \
  X(101, TiSi2, Silicide, 4.0, true)                                           \
  X(102, MoSi2, Silicide, 6.3, true)                                           \
  /* Compound semiconductors */                                                \
  X(110, Ge, Compound, 5.32, false)                                            \
  X(111, GaN, Compound, 6.15, false)                                           \
  X(112, GaAs, Compound, 5.32, false)                                          \
  X(113, InP, Compound, 4.81, false)                                           \
  X(114, InGaAs, Compound, 5.6, false)                                         \
  X(115, SiGaN, Compound, 5.0, false)                                          \
  X(116, SiOCH, OxideNitride, 1.8, false)                                      \
  /* 2D + emerging */                                                          \
  X(130, Graphene, TwoD, 2.2, true)                                            \
  X(131, MoS2, TwoD, 5.0, false)                                               \
  X(132, WS2, TwoD, 7.5, false)                                                \
  X(133, WSe2, TwoD, 9.3, false)                                               \
  X(134, VO2, TwoD, 4.6, false)                                                \
  X(135, GST, TwoD, 6.1, false)                                                \
  /* Transparent conductors */                                                 \
  X(150, ITO, TCO, 7.1, true)                                                  \
  X(151, ZnO, TCO, 5.6, true)                                                  \
  X(152, AZO, TCO, 5.5, true)                                                  \
  /* Misc hardmask aliases */                                                  \
  X(170, SiON_HM, Hardmask, 2.4, false)                                        \
  X(171, SiN_HM, Hardmask, 3.1, false)                                         \
  X(172, SiC_HM, Hardmask, 3.2, false)                                         \
  X(173, TiO, Misc, 4.9, false)                                                \
  X(174, ZrO, Misc, 5.2, false)                                                \
  X(175, SiO2_HM, Hardmask, 2.2, false)

enum class Material : uint16_t {
#define ENUM_ELEM(id, sym, cat, dens, cond) sym = id,
  MATERIAL_LIST(ENUM_ELEM)
#undef ENUM_ELEM
};

struct MaterialInfo {
  std::string_view name;
  MaterialCategory category;
  double density_gcm3; // nominal; tune per PDK
  bool conductive;
};

constexpr uint16_t kMaterialMaxId = 175;

constexpr std::array<MaterialInfo, kMaterialMaxId + 1> kMaterialTable = [] {
  std::array<MaterialInfo, kMaterialMaxId + 1> t{};
  // default fill
  for (auto &e : t)
    e = {"Undefined", MaterialCategory::Generic, 0.0, false};
    // populate
#define FILL_ROW(id, sym, cat, dens, cond)                                     \
  t[id] = {#sym, MaterialCategory::cat, dens, cond};
  MATERIAL_LIST(FILL_ROW)
#undef FILL_ROW
  return t;
}();

constexpr const MaterialInfo &info(Material m) {
  auto id = static_cast<uint16_t>(m);
  return kMaterialTable[id <= kMaterialMaxId ? id : 0]; // 0 == Mask or change
                                                        // to Undefined id
}

constexpr std::string_view to_string_view(Material m) { return info(m).name; }
constexpr MaterialCategory categoryOf(Material m) { return info(m).category; }
constexpr double density(Material m) { return info(m).density_gcm3; }
constexpr bool isConductive(Material m) { return info(m).conductive; }

#ifndef __CUDACC__
/// A class that wraps the viennals MaterialMap class and provides a more user
/// friendly interface. It also provides a mapping from the integer material id
/// to the Material enum.
class MaterialMap {
  SmartPointer<viennals::MaterialMap> map_;

public:
  MaterialMap() { map_ = SmartPointer<viennals::MaterialMap>::New(); };

  void insertNextMaterial(Material material = Material::Undefined) {
    map_->insertNextMaterial(static_cast<int>(material));
  }

  // Returns the material at the given index. If the index is out of bounds, it
  // returns Material::GAS.
  [[nodiscard]] Material getMaterialAtIdx(std::size_t idx) const {
    if (idx >= size())
      return Material::GAS;
    int matId = map_->getMaterialId(idx);
    return mapToMaterial(matId);
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

  static inline Material mapToMaterial(const int matId) {
    return static_cast<Material>(matId);
  }

  template <class T> static inline Material mapToMaterial(const T matId) {
    return mapToMaterial(static_cast<int>(matId));
  }

  template <class T>
  static inline bool isMaterial(const T matId, const Material material) {
    return mapToMaterial(matId) == material;
  }

  template <class T> static inline bool isHardmask(const T matId) {
    return categoryOf(mapToMaterial(matId)) == MaterialCategory::Hardmask;
  }

  static inline std::string toString(const Material matId) {
    return std::string(to_string_view(matId));
  }

  static inline std::string toString(const int matId) {
    return std::string(to_string_view(mapToMaterial(matId)));
  }
};
#endif

} // namespace viennaps
