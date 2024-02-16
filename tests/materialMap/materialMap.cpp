#include "psMaterials.hpp"
#include <cassert>

int main() {
  // Constructor test
  psMaterialMap materialMap;
  assert(materialMap.size() == 0);

  // InsertNextMaterial test
  materialMap.insertNextMaterial(psMaterial::Si);
  assert(materialMap.size() == 1);
  assert(materialMap.getMaterialAtIdx(0) == psMaterial::Si);

  // GetMaterialAtIdx test
  materialMap.insertNextMaterial(psMaterial::SiO2);
  assert(materialMap.getMaterialAtIdx(0) == psMaterial::Si);
  assert(materialMap.getMaterialAtIdx(1) == psMaterial::SiO2);
  assert(materialMap.getMaterialAtIdx(2) == psMaterial::GAS);

  // SetMaterialAtIdx test
  materialMap.setMaterialAtIdx(0, psMaterial::SiO2);
  assert(materialMap.getMaterialAtIdx(0) == psMaterial::SiO2);

  // MapToMaterial test
  assert(psMaterialMap::mapToMaterial(1) == psMaterial::Si);
  assert(psMaterialMap::mapToMaterial(2) == psMaterial::SiO2);
  assert(psMaterialMap::mapToMaterial(19) == psMaterial::None);

  // IsMaterial test
  assert(psMaterialMap::isMaterial(1, psMaterial::Si));
  assert(!psMaterialMap::isMaterial(1, psMaterial::SiO2));

  return 0;
}