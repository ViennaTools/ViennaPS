#pragma once

#include <lsMaterialMap.hpp>

#include <vcLogger.hpp>
#include <vcSmartPointer.hpp>

#include <string>

namespace viennaps {

using namespace viennacore;

enum class Material : int {
  Undefined = -1,
  Mask = 0,
  Si = 1,
  SiO2 = 2,
  Si3N4 = 3,
  SiN = 4,
  SiON = 5,
  SiC = 6,
  SiGe = 7,
  PolySi = 8,
  GaN = 9,
  W = 10,
  Al2O3 = 11,
  HfO2 = 12,
  TiN = 13,
  Cu = 14,
  Polymer = 15,
  Dielectric = 16,
  Metal = 17,
  Air = 18,
  GAS = 19
};

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
      Logger::getInstance()
          .addError("Setting material with out-of-bounds index.")
          .print();
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
    if (matId > 19 || matId < -1)
      return Material::Undefined;
    return static_cast<Material>(matId);
  }

  template <class T> static inline Material mapToMaterial(const T matId) {
    return mapToMaterial(static_cast<int>(matId));
  }

  template <class T>
  static inline bool isMaterial(const T matId, const Material material) {
    return mapToMaterial(matId) == material;
  }

  template <class T> static inline std::string getMaterialName(const T matId) {
    switch (auto material = mapToMaterial(matId)) {
    case Material::Undefined:
      return "Undefined";
    case Material::Mask:
      return "Mask";
    case Material::Si:
      return "Si";
    case Material::SiO2:
      return "SiO2";
    case Material::Si3N4:
      return "Si3N4";
    case Material::SiN:
      return "SiN";
    case Material::SiON:
      return "SiON";
    case Material::SiC:
      return "SiC";
    case Material::SiGe:
      return "SiGe";
    case Material::PolySi:
      return "PolySi";
    case Material::GaN:
      return "GaN";
    case Material::W:
      return "W";
    case Material::Al2O3:
      return "Al2O3";
    case Material::HfO2:
      return "HfO2";
    case Material::TiN:
      return "TiN";
    case Material::Cu:
      return "Cu";
    case Material::Polymer:
      return "Polymer";
    case Material::Dielectric:
      return "Dielectric";
    case Material::Metal:
      return "Metal";
    case Material::Air:
      return "Air";
    case Material::GAS:
      return "GAS";
    default:
      return "Unknown";
    }
  }
};

} // namespace viennaps
