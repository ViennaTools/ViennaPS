#include <materials/psMaterialValueMap.hpp>

#include <algorithm>
#include <vector>

using namespace viennaps;

void testBuiltInMaterial() {
  MaterialValueMap<int> map;
  map.setDefault(-1);
  map.set(Material::Si, 1);
  map.set(Material::SiO2, 2);

  assert(map.get(Material::Si) == 1);
  assert(map.get(Material::SiO2) == 2);
  assert(map.get(Material::Si3N4) == -1); // default
}

void testCustomMaterial() {
  MaterialValueMap<int> map;
  map.setDefault(0);
  Material customMat = MaterialMap::fromString("custom");
  map.set(customMat, 3);

  assert(map.get(customMat) == 3);
  assert(map.get(Material::Si) == 0); // default

  Material customMat2 = MaterialMap::fromString("custom2");
  assert(map.get(customMat2) == 0); // default

  // test operator[]
  map[customMat2] = 4;
  assert(map.get(customMat2) == 4);
}

void testIteration() {
  MaterialValueMap<int> map;
  map.set(Material::Si, 1);
  map.set(Material::SiO2, 2);
  map.set(Material::custom(42), 7);

  std::vector<std::pair<Material, int>> expected = {
      {Material::Si, 1},
      {Material::SiO2, 2},
      {Material::custom(42), 7},
  };

  std::vector<std::pair<Material, int>> actual;
  for (auto entry : map) {
    actual.emplace_back(entry.material, entry.value);
  }

  assert(actual.size() == expected.size());
  for (const auto &item : expected) {
    assert(std::find(actual.begin(), actual.end(), item) != actual.end());
  }
}

int main() {
  testBuiltInMaterial();
  testCustomMaterial();
  testIteration();
  return 0;
}