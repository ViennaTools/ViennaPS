#include "psMaterials.hpp"
#include <cassert>

using namespace viennaps;

int main() {
  // Constructor test
  MaterialMap materialMap;
  assert(materialMap.size() == 0);

  // InsertNextMaterial test
  materialMap.insertNextMaterial(Material::Si);
  assert(materialMap.size() == 1);
  assert(materialMap.getMaterialAtIdx(0) == Material::Si);

  // GetMaterialAtIdx test
  materialMap.insertNextMaterial(Material::SiO2);
  assert(materialMap.getMaterialAtIdx(0) == Material::Si);
  assert(materialMap.getMaterialAtIdx(1) == Material::SiO2);
  assert(materialMap.getMaterialAtIdx(2) == Material::GAS);

  // SetMaterialAtIdx test
  materialMap.setMaterialAtIdx(0, Material::SiO2);
  assert(materialMap.getMaterialAtIdx(0) == Material::SiO2);

  // MapToMaterial test
  assert(MaterialMap::mapToMaterial(1) == Material::Si);
  assert(MaterialMap::mapToMaterial(2) == Material::SiO2);
  assert(MaterialMap::mapToMaterial(20) == Material::None);

  // IsMaterial test
  assert(MaterialMap::isMaterial(1, Material::Si));
  assert(!MaterialMap::isMaterial(1, Material::SiO2));

  return 0;
}