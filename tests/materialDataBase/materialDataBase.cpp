#include <materials/psMaterialDataBase.hpp>

#include <vcTestAsserts.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>

using namespace viennaps;

void testWriteAndReadRoundTrip() {
  auto &registry = MaterialRegistry::instance();

  MaterialDataBase db;

  materials::DBEntry siEntry;
  siEntry.category = MaterialCategory::Silicon;
  siEntry.density_gcm3 = 2.9;
  siEntry.density_acm3 = 4.2e22;
  db.setEntry(Material::Si, siEntry);

  const auto customMaterial = registry.registerMaterial("MaterialDBCustom");
  materials::DBEntry customEntry;
  customEntry.category = MaterialCategory::Compound;
  customEntry.density_gcm3 = 5.6;
  customEntry.density_acm3 = std::numeric_limits<double>::quiet_NaN();
  db.setEntry(customMaterial, customEntry);

  const std::filesystem::path filePath = "test_material_database.json";
  if (std::filesystem::exists(filePath)) {
    std::filesystem::remove(filePath);
  }

  db.writeToFile(filePath);
  VC_TEST_ASSERT(std::filesystem::exists(filePath));

  MaterialDataBase loaded;
  loaded.readFromFile(filePath);

  VC_TEST_ASSERT(loaded.hasEntry(Material::Si));
  VC_TEST_ASSERT(loaded.hasEntry(customMaterial));

  const auto loadedSi = loaded.getEntry(Material::Si);
  VC_TEST_ASSERT(loadedSi.category == MaterialCategory::Silicon);
  VC_TEST_ASSERT(loadedSi.density_gcm3 == 2.9);
  VC_TEST_ASSERT(loadedSi.density_acm3 == 4.2e22);

  const auto loadedCustom = loaded.getEntry(customMaterial);
  VC_TEST_ASSERT(loadedCustom.category == MaterialCategory::Compound);
  VC_TEST_ASSERT(loadedCustom.density_gcm3 == 5.6);
  VC_TEST_ASSERT(std::isnan(loadedCustom.density_acm3));

  std::filesystem::remove(filePath);
}

void testInvalidSchemaVersionThrows() {
  const std::filesystem::path filePath = "test_material_database_invalid.json";
  {
    std::ofstream out(filePath, std::ios::trunc);
    out << R"({"schemaVersion": 999, "materials": {}})";
  }

  MaterialDataBase db;
  bool didThrow = false;
  try {
    db.readFromFile(filePath);
  } catch (const std::exception &) {
    didThrow = true;
  }

  VC_TEST_ASSERT(didThrow);
  std::filesystem::remove(filePath);
}

int main() {
  testWriteAndReadRoundTrip();
  testInvalidSchemaVersionThrows();
  return 0;
}
