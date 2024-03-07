#pragma once

#include <lsMaterialMap.hpp>
#include <psLogger.hpp>
#include <psSmartPointer.hpp>

enum class psMaterial : int {
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
/// to the psMaterial enum.
class psMaterialMap {
  psSmartPointer<lsMaterialMap> map;

public:
  psMaterialMap() { map = psSmartPointer<lsMaterialMap>::New(); };

  void insertNextMaterial(psMaterial material = psMaterial::None) {
    map->insertNextMaterial(static_cast<int>(material));
  }

  // Returns the material at the given index. If the index is out of bounds, it
  // returns psMaterial::GAS.
  psMaterial getMaterialAtIdx(std::size_t idx) const {
    if (idx >= size())
      return psMaterial::GAS;
    int matId = map->getMaterialId(idx);
    return mapToMaterial(matId);
  }

  void setMaterialAtIdx(std::size_t idx, const psMaterial material) {
    if (idx >= size()) {
      psLogger::getInstance()
          .addWarning("Setting material with out-of-bounds index.")
          .print();
    }
    map->setMaterialId(idx, static_cast<int>(material));
  }

  psSmartPointer<lsMaterialMap> getMaterialMap() const { return map; }

  inline std::size_t size() const { return map->getNumberOfLayers(); }

  static inline psMaterial mapToMaterial(const int matId) {
    if (matId > 18 || matId < -1)
      return psMaterial::None;
    return static_cast<psMaterial>(matId);
  }

  template <class T> static inline psMaterial mapToMaterial(const T matId) {
    return mapToMaterial(static_cast<int>(matId));
  }

  template <class T>
  static inline bool isMaterial(const T matId, const psMaterial material) {
    return mapToMaterial(matId) == material;
  }
};
