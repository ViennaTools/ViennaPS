#include <geometries/psMakeTrench.hpp>
#include <models/psDirectionalProcess.hpp>
#include <models/psIsotropicProcess.hpp>
#include <process/psProcess.hpp>
#include <psDomain.hpp>

#include <vcTestAsserts.hpp>

using namespace viennaps;

void testMaterialRegistry() {
  auto &registry = MaterialRegistry::instance();

  const auto builtIn = registry.registerMaterial("Si");
  VC_TEST_ASSERT(builtIn.isBuiltIn());
  VC_TEST_ASSERT(builtIn == Material::Si);
  VC_TEST_ASSERT(registry.customMaterialCount() == 0);

  const auto customA = registry.registerMaterial("CustomFoo");
  const auto customARepeat = registry.registerMaterial("CustomFoo");
  const auto customB = registry.registerMaterial("CustomBar");
  VC_TEST_ASSERT(registry.customMaterialCount() == 2);

  VC_TEST_ASSERT(customA.isCustom());
  VC_TEST_ASSERT(customA == customARepeat);
  VC_TEST_ASSERT(customA != customB);
  VC_TEST_ASSERT(registry.hasMaterial("CustomFoo"));
  VC_TEST_ASSERT(registry.getMaterial("CustomFoo") == customA);
  VC_TEST_ASSERT(registry.getName(customA) == "CustomFoo");
  VC_TEST_ASSERT(registry.customMaterialCount() == 2);

  const auto maybeFound = registry.findMaterial("CustomBar");
  VC_TEST_ASSERT(maybeFound.has_value());
  VC_TEST_ASSERT(maybeFound.value() == customB);

  const auto maybeMissing = registry.findMaterial("DoesNotExist");
  VC_TEST_ASSERT(!maybeMissing.has_value());
}

void testCustomMaterialMap() {
  Logger::setLogLevel(LogLevel::ERROR);

  auto domain =
      Domain<double, 2>::New(0.97, 50.0, BoundaryType::REFLECTIVE_BOUNDARY);
  MakeTrench<double, 2>(domain, 15.0, 0.0, 0.0, 3.0).apply();

  auto isoEtch =
      SmartPointer<IsotropicProcess<double, 2>>::New(-1.0, Material::Mask);
  Process<double, 2>(domain, isoEtch, 10.0).apply();

  // domain->saveSurfaceMesh("testCustomMaterialMap_0");

  domain->duplicateTopLevelSet("CustomMaterial");

  auto isoDep = SmartPointer<IsotropicProcess<double, 2>>::New(1.0);
  Process<double, 2>(domain, isoDep, 2.0).apply();

  // domain->saveSurfaceMesh("testCustomMaterialMap_1");

  auto customMaterial = MaterialMap::fromString("CustomMaterial");
  VC_TEST_ASSERT(customMaterial ==
                 MaterialRegistry::instance().getMaterial("CustomMaterial"));
  VC_TEST_ASSERT(customMaterial ==
                 domain->getMaterialMap()->getMaterialAtIdx(2));
  std::unordered_map<Material, std::pair<double, double>> materialRates;
  materialRates[Material::Mask] = {0.0, 0.0};
  materialRates[customMaterial] = {1.0, 0.0};
  materialRates[Material::Si] = {0.5, -0.5};

  auto directionalProcess = SmartPointer<DirectionalProcess<double, 2>>::New(
      Vec3Dd{0.0, -1.0, 0.0}, materialRates);
  Process<double, 2>(domain, directionalProcess, 5.0).apply();

  // domain->saveSurfaceMesh("testCustomMaterialMap_2");
}

int main() {

  testMaterialRegistry();
  testCustomMaterialMap();

  return 0;
}
