#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>

#include <lsAdvect.hpp>
#include <lsDomain.hpp>
#include <lsExpand.hpp>

#include <geometries/psMakeHole.hpp>
#include <geometries/psMakeTrench.hpp>
#include <psDomain.hpp>
#include <psMaterials.hpp>
#include <psReader.hpp>
#include <psWriter.hpp>

// Test helper function to check if two values are approximately equal
template <typename T> bool approxEqual(T a, T b, T epsilon = 1e-10) {
  return std::abs(a - b) < epsilon;
}

// Test helper function to verify domain equality
template <class T, int D>
bool domainsEqual(viennaps::SmartPointer<viennaps::Domain<T, D>> domainA,
                  viennaps::SmartPointer<viennaps::Domain<T, D>> domainB) {

  // Check if both domains have same number of level sets
  auto &lsA = domainA->getLevelSets();
  auto &lsB = domainB->getLevelSets();

  if (lsA.size() != lsB.size()) {
    std::cout << "Different number of level sets: " << lsA.size() << " vs "
              << lsB.size() << std::endl;
    return false;
  }

  // Check material maps
  auto &mapA = domainA->getMaterialMap();
  auto &mapB = domainB->getMaterialMap();

  if ((mapA == nullptr) != (mapB == nullptr)) {
    std::cout << "Material map existence mismatch" << std::endl;
    return false;
  }

  if (mapA != nullptr && mapB != nullptr) {
    if (mapA->size() != mapB->size()) {
      std::cout << "Different number of materials: " << mapA->size() << " vs "
                << mapB->size() << std::endl;
      return false;
    }

    // Check each material
    for (size_t i = 0; i < mapA->size(); i++) {
      if (mapA->getMaterialAtIdx(i) != mapB->getMaterialAtIdx(i)) {
        std::cout << "Material mismatch at index " << i << ": "
                  << static_cast<int>(mapA->getMaterialAtIdx(i)) << " vs "
                  << static_cast<int>(mapB->getMaterialAtIdx(i)) << std::endl;
        return false;
      }
    }
  }

  // Check grid delta
  if (!approxEqual(domainA->getGridDelta(), domainB->getGridDelta())) {
    std::cout << "Grid delta mismatch: " << domainA->getGridDelta() << " vs "
              << domainB->getGridDelta() << std::endl;
    return false;
  }

  // We consider the domains equal if they have:
  // 1. Same number of level sets
  // 2. Same material map
  // 3. Same grid delta
  // For a more thorough test, we could also check the level set values

  return true;
}

// Test function for 3D hole write and read
template <class T> bool test3DHoleWriteAndRead() {
  constexpr int D = 3;
  using DomainType = viennaps::SmartPointer<viennaps::Domain<T, D>>;

  std::cout << "\n=========================================" << std::endl;
  std::cout << "Testing 3D hole geometry write and read..." << std::endl;

  // Create a test filename
  std::string testFileName = "testHole3D.vpsd";

  // Remove any existing test file
  if (std::filesystem::exists(testFileName)) {
    std::filesystem::remove(testFileName);
  }

  // Create a domain for testing
  const T gridDelta = 1.0;
  const T xExtent = 100.0;
  const T yExtent = 100.0;
  const T zExtent = 80.0;
  const T holeRadius = 25.0;
  const T holeDepth = 40.0;
  const T taperAngle = 5.0; // 5 degree taper

  // Create a 3D domain
  DomainType domain = DomainType::New(gridDelta, xExtent, yExtent);

  // Apply the hole geometry
  viennaps::MakeHole<T, D>(domain, holeRadius, holeDepth, taperAngle,
                           10.0,                      // maskHeight
                           2.0,                       // maskTaperAngle
                           viennaps::HoleShape::FULL, // full hole
                           viennaps::Material::Si, viennaps::Material::SiO2)
      .apply();

  // Expand the level sets for better visualization
  for (auto &ls : domain->getLevelSets()) {
    viennals::Expand<T, D>(ls, 2).apply();
  }

  // Write the domain to file
  viennaps::Writer<T, D> writer(domain, testFileName);
  writer.apply();

  // Write meshes for visualization
  domain->saveSurfaceMesh("writtenHole3D-surfaceMesh");
  domain->saveLevelSetMesh("writtenHole3D-levelSetMesh", 5);

  // Verify the file was created
  if (!std::filesystem::exists(testFileName)) {
    std::cerr << "Error: 3D hole test file was not created!" << std::endl;
    return false;
  }

  // Read the domain back from file
  viennaps::Reader<T, D> reader(testFileName);
  auto readDomain = reader.apply();

  // Write meshes for visualization
  readDomain->saveSurfaceMesh("readHole3D-surfaceMesh");
  readDomain->saveLevelSetMesh("readHole3D-levelSetMesh", 5);

  // Verify the domains are equal
  if (domainsEqual(domain, readDomain)) {
    std::cout << "✓ 3D hole domain successfully written and read back!"
              << std::endl;
  } else {
    std::cerr << "✗ Error: Read hole domain does not match original domain!"
              << std::endl;
    return false;
  }

  // Print domain information for visual verification
  std::cout << "\nOriginal 3D hole domain info:" << std::endl;
  std::cout << "Number of level sets: " << domain->getLevelSets().size()
            << std::endl;
  std::cout << "Grid delta: " << domain->getGridDelta() << std::endl;

  std::cout << "\nRead 3D hole domain info:" << std::endl;
  std::cout << "Number of level sets: " << readDomain->getLevelSets().size()
            << std::endl;
  std::cout << "Grid delta: " << readDomain->getGridDelta() << std::endl;

  return true;
}

int main() {
  constexpr int D = 2;
  using NumericType = double;
  using DomainType = viennaps::SmartPointer<viennaps::Domain<NumericType, D>>;

  std::cout << "Testing psWriter and psReader..." << std::endl;

  // Create a test filename
  std::string testFileName = "testDomain.vpsd";

  // Remove any existing test file
  if (std::filesystem::exists(testFileName)) {
    std::filesystem::remove(testFileName);
  }

  // Create a domain for testing
  const NumericType gridDelta = 1.0;
  const NumericType xExtent = 100.0;
  const NumericType yExtent = 100.0;
  const NumericType trenchWidth = 50.0;
  const NumericType trenchDepth = 40.0;
  const NumericType taperAngle = 5.0; // 5 degree taper

  // Create a trench domain
  DomainType domain = DomainType::New(gridDelta, xExtent, yExtent);

  // Apply the trench geometry
  viennaps::MakeTrench<NumericType, D>(
      domain, trenchWidth, trenchDepth, taperAngle,
      10.0,  // maskHeight
      2.0,   // maskTaperAngle
      false, // halfTrench
      viennaps::Material::Si, viennaps::Material::SiO2)
      .apply();

  // Expand the level sets for better visualization
  for (auto &ls : domain->getLevelSets()) {
    viennals::Expand<NumericType, D>(ls, 2).apply();
  }

  // Write the domain to file
  viennaps::Writer<NumericType, D> writer(domain, testFileName);
  writer.apply();

  // Write meshes for visualization
  domain->saveSurfaceMesh("writtenDomain-surfaceMesh");
  domain->saveLevelSetMesh("writtenDomain-levelSetMesh", 5);

  // Verify the file was created
  if (!std::filesystem::exists(testFileName)) {
    std::cerr << "Error: Test file was not created!" << std::endl;
    return 1;
  }

  // Read the domain back from file
  viennaps::Reader<NumericType, D> reader(testFileName);
  auto readDomain = reader.apply();

  // Write meshes for visualization
  readDomain->saveSurfaceMesh("readDomain-surfaceMesh");
  readDomain->saveLevelSetMesh("readDomain-levelSetMesh", 5);

  // Verify the domains are equal
  if (domainsEqual(domain, readDomain)) {
    std::cout << "✓ Domain successfully written and read back!" << std::endl;
  } else {
    std::cerr << "✗ Error: Read domain does not match original domain!"
              << std::endl;
    return 1;
  }

  // Print domain information for visual verification
  std::cout << "\nOriginal domain info:" << std::endl;
  std::cout << "Number of level sets: " << domain->getLevelSets().size()
            << std::endl;
  std::cout << "Grid delta: " << domain->getGridDelta() << std::endl;

  std::cout << "\nRead domain info:" << std::endl;
  std::cout << "Number of level sets: " << readDomain->getLevelSets().size()
            << std::endl;
  std::cout << "Grid delta: " << readDomain->getGridDelta() << std::endl;

  std::cout << "2D test completed successfully!" << std::endl;

  // Test the 3D hole write and read functionality
  bool hole3DTestPassed = test3DHoleWriteAndRead<NumericType>();

  if (!hole3DTestPassed) {
    std::cerr << "3D hole test failed!" << std::endl;
    return 1;
  }

  std::cout << "\nAll tests completed successfully!" << std::endl;
  return 0;
}
