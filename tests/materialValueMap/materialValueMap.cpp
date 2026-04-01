#include <materials/psMaterialValueMap.hpp>
#include <vcTestAsserts.hpp>

#include <algorithm>
#include <vector>

using namespace viennaps;

void testBuiltInMaterial() {
  MaterialValueMap<int> map;
  map.setDefault(-1);
  map.set(Material::Si, 1);
  map.set(Material::SiO2, 2);

  VC_TEST_ASSERT(map.get(Material::Si) == 1);
  VC_TEST_ASSERT(map.get(Material::SiO2) == 2);
  VC_TEST_ASSERT(map.get(Material::Si3N4) == -1); // default
}

void testCustomMaterial() {
  MaterialValueMap<int> map;
  map.setDefault(0);
  Material customMat = MaterialMap::fromString("custom");
  map.set(customMat, 3);

  VC_TEST_ASSERT(map.get(customMat) == 3);
  VC_TEST_ASSERT(map.get(Material::Si) == 0); // default

  Material customMat2 = MaterialMap::fromString("custom2");
  VC_TEST_ASSERT(map.get(customMat2) == 0); // default

  // test operator[]
  map[customMat2] = 4;
  VC_TEST_ASSERT(map.get(customMat2) == 4);
}

void testIteration() {
  MaterialValueMap<int> map;
  auto customMat = MaterialMap::fromString("custom");
  map.set(Material::Si, 1);
  map.set(Material::SiO2, 2);
  map.set(customMat, 7);

  std::vector<std::pair<Material, int>> expected = {
      {Material::Si, 1},
      {Material::SiO2, 2},
      {customMat, 7},
  };

  size_t idx = 0;
  for (const auto &[material, value] : map) {
    VC_TEST_ASSERT(idx < expected.size());
    VC_TEST_ASSERT(material == expected[idx].first);
    VC_TEST_ASSERT(value == expected[idx].second);
    std::cout << "Material: " << MaterialMap::toString(material) << "("
              << material.legacyId() << ")" << ", Value: " << value
              << std::endl;
    ++idx;
  }

  VC_TEST_ASSERT(map.getEntryByIndex(0) == 1);
  VC_TEST_ASSERT(map.getEntryByIndex(1) == 2);
  VC_TEST_ASSERT(map.getEntryByIndex(2) == 7);
}

int main() {
  testBuiltInMaterial();
  testCustomMaterial();
  testIteration();
  return 0;
}