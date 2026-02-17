#include "psMaterials.hpp"

#include <vcTestAsserts.hpp>

using namespace viennaps;

int main() {
  // Constructor test
  MaterialMap materialMap;
  VC_TEST_ASSERT(materialMap.size() == 0);

  // InsertNextMaterial test
  materialMap.insertNextMaterial(Material::Si);
  VC_TEST_ASSERT(materialMap.size() == 1);
  VC_TEST_ASSERT(materialMap.getMaterialAtIdx(0) == Material::Si);

  // GetMaterialAtIdx test
  materialMap.insertNextMaterial(Material::SiO2);
  VC_TEST_ASSERT(materialMap.getMaterialAtIdx(0) == Material::Si);
  VC_TEST_ASSERT(materialMap.getMaterialAtIdx(1) == Material::SiO2);
  VC_TEST_ASSERT(materialMap.getMaterialAtIdx(2) == Material::Undefined);

  materialMap.removeMaterial();
  VC_TEST_ASSERT(materialMap.size() == 1);

  // SetMaterialAtIdx test
  materialMap.setMaterialAtIdx(0, Material::SiO2);
  VC_TEST_ASSERT(materialMap.getMaterialAtIdx(0) == Material::SiO2);

  VC_TEST_ASSERT(MaterialMap::isValidMaterial(Material::Si));
  VC_TEST_ASSERT(!MaterialMap::isValidMaterial(static_cast<Material>(999)));

  return 0;
}