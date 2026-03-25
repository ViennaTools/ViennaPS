#include "materials/psMaterialValueMap.hpp"
#include "materials/psMaterials.hpp"

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

  // Built-in mapping checks
  VC_TEST_ASSERT(builtInMaterialToString(BuiltInMaterial::Si) == "Si");
  VC_TEST_ASSERT(builtInMaterialFromString("SiO2") == BuiltInMaterial::SiO2);
  VC_TEST_ASSERT(Material::Si.isBuiltIn());

  // Custom material registration checks
  MaterialRegistry registry;
  const auto customA = registry.registerMaterial("MyCustom");
  const auto customA2 = registry.registerMaterial("MyCustom");
  const auto customB = registry.registerMaterial("AnotherCustom");
  VC_TEST_ASSERT(customA.isCustom());
  VC_TEST_ASSERT(customA == customA2);
  VC_TEST_ASSERT(customA != customB);
  VC_TEST_ASSERT(registry.getName(customA) == "MyCustom");
  VC_TEST_ASSERT(registry.getMaterial("MyCustom") == customA);

  // MaterialMap supports custom IDs through legacy mapping.
  materialMap.insertNextMaterial(customA);
  VC_TEST_ASSERT(materialMap.getMaterialAtIdx(1) == customA);

  // Mixed MaterialValueMap support.
  MaterialValueMap<double> mixedRates;
  mixedRates.set(Material::Si, 1.0);
  mixedRates.set(customA, 2.0);
  VC_TEST_ASSERT(mixedRates.get(Material::Si) == 1.0);
  VC_TEST_ASSERT(mixedRates.get(customA) == 2.0);

  VC_TEST_ASSERT(MaterialMap::isValidMaterial(Material::Si));
  VC_TEST_ASSERT(MaterialMap::isValidMaterial(customA));

  return 0;
}