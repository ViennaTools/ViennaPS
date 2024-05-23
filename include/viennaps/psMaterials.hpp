#pragma once

#include <lsMaterialMap.hpp>

#include <vcLogger.hpp>
#include <vcSmartPointer.hpp>

namespace viennaps {

using namespace viennacore;

enum class Material : int {
  None = -1,
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
  TiN = 12,
  Cu = 13,
  Polymer = 14,
  Dielectric = 15,
  Metal = 16,
  Air = 17,
  GAS = 18
};

/// A class that wraps the lsMaterialMap class and provides a more user
/// friendly interface. It also provides a mapping from the integer material id
/// to the Material enum.
class MaterialMap {
  SmartPointer<lsMaterialMap> map_;

public:
  MaterialMap() { map_ = SmartPointer<lsMaterialMap>::New(); };

  void insertNextMaterial(Material material = Material::None) {
    map_->insertNextMaterial(static_cast<int>(material));
  }

  // Returns the material at the given index. If the index is out of bounds, it
  // returns Material::GAS.
  Material getMaterialAtIdx(std::size_t idx) const {
    if (idx >= size())
      return Material::GAS;
    int matId = map_->getMaterialId(idx);
    return mapToMaterial(matId);
  }

  void setMaterialAtIdx(std::size_t idx, const Material material) {
    if (idx >= size()) {
      Logger::getInstance()
          .addWarning("Setting material with out-of-bounds index.")
          .print();
    }
    map_->setMaterialId(idx, static_cast<int>(material));
  }

  SmartPointer<lsMaterialMap> getMaterialMap() const { return map_; }

  inline std::size_t const size() const { return map_->getNumberOfLayers(); }

  static inline Material mapToMaterial(const int matId) {
    if (matId > 18 || matId < -1)
      return Material::None;
    return static_cast<Material>(matId);
  }

  template <class T> static inline Material mapToMaterial(const T matId) {
    return mapToMaterial(static_cast<int>(matId));
  }

  template <class T>
  static inline bool isMaterial(const T matId, const Material material) {
    return mapToMaterial(matId) == material;
  }
};

} // namespace viennaps
