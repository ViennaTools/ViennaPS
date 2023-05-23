#pragma once

#include <lsMaterialMap.hpp>
#include <psSmartPointer.hpp>

enum class psMaterial : int {
  Undefined = -1,
  Mask = 0,
  Si = 1,
  SiO2 = 2,
  Si3N4 = 3,
  PolySi = 4,
  Tungsten = 5,
  Al2O3 = 6,
  Polymer = 7,
  Dielectric = 8,
  Metal = 9
};

class psMaterialMap {
  psSmartPointer<lsMaterialMap> map;

public:
  psMaterialMap() { map = psSmartPointer<lsMaterialMap>::New(); };

  void insertNextMaterial(psMaterial material = psMaterial::Undefined) {
    map->insertNextMaterial(static_cast<int>(material));
  }

  psMaterial getMaterialAtIdx(std::size_t idx) const {
    int matId = map->getMaterialId(idx);
    return mapToMaterial(matId);
  }

  psSmartPointer<lsMaterialMap> getMaterialMap() const { return map; }

  std::size_t size() const { return map->getNumberOfLayers(); }

  static inline psMaterial mapToMaterial(const int matId) {
    if (matId > 9 || matId < -1)
      return psMaterial::Undefined;
    return static_cast<psMaterial>(matId);
  }
};
