#include "materials/psMaterials.hpp"

#include <vcTestAsserts.hpp>

using namespace viennaps;

int main() {
  MaterialRegistry registry;

  const auto builtIn = registry.registerMaterial("Si");
  VC_TEST_ASSERT(builtIn.isBuiltIn());
  VC_TEST_ASSERT(builtIn == Material::Si);

  const auto customA = registry.registerMaterial("CustomFoo");
  const auto customARepeat = registry.registerMaterial("CustomFoo");
  const auto customB = registry.registerMaterial("CustomBar");

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

  return 0;
}
