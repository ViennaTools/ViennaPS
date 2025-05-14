#include <cassert>
#include <iostream>
#include <string>
#include <cmath>
#include <filesystem>

#include <lsAdvect.hpp>
#include <lsDomain.hpp>
#include <lsExpand.hpp>

#include <psDomain.hpp>
#include <psWriter.hpp>
#include <psReader.hpp>
#include <psMaterials.hpp>
#include <geometries/psMakeTrench.hpp>

// Test helper function to check if two values are approximately equal
template<typename T>
bool approxEqual(T a, T b, T epsilon = 1e-10) {
    return std::abs(a - b) < epsilon;
}

// Test helper function to verify domain equality
template<class T, int D>
bool domainsEqual(viennaps::SmartPointer<viennaps::Domain<T, D>> domainA, 
                  viennaps::SmartPointer<viennaps::Domain<T, D>> domainB) {
    
    // Check if both domains have same number of level sets
    auto& lsA = domainA->getLevelSets();
    auto& lsB = domainB->getLevelSets();
    
    if (lsA.size() != lsB.size()) {
        std::cout << "Different number of level sets: " << lsA.size() << " vs " << lsB.size() << std::endl;
        return false;
    }
    
    // Check material maps
    auto& mapA = domainA->getMaterialMap();
    auto& mapB = domainB->getMaterialMap();
    
    if ((mapA == nullptr) != (mapB == nullptr)) {
        std::cout << "Material map existence mismatch" << std::endl;
        return false;
    }
    
    if (mapA != nullptr && mapB != nullptr) {
        if (mapA->size() != mapB->size()) {
            std::cout << "Different number of materials: " << mapA->size() << " vs " << mapB->size() << std::endl;
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
        std::cout << "Grid delta mismatch: " << domainA->getGridDelta() << " vs " << domainB->getGridDelta() << std::endl;
        return false;
    }
    
    // We consider the domains equal if they have:
    // 1. Same number of level sets
    // 2. Same material map
    // 3. Same grid delta
    // For a more thorough test, we could also check the level set values
    
    return true;
}

int main() {
    constexpr int D = 2;
    using NumericType = double;
    using DomainType = viennaps::SmartPointer<viennaps::Domain<NumericType, D>>;
    
    std::cout << "Testing psWriter and psReader..." << std::endl;
    
    // Create a test filename
    std::string testFileName = "testDomain.psd";
    
    // Remove any existing test file
    if (std::filesystem::exists(testFileName)) {
        std::filesystem::remove(testFileName);
    }
    
    // Create a domain for testing
    const NumericType gridDelta = 0.2;
    const NumericType xExtent = 10.0;
    const NumericType yExtent = 10.0;
    const NumericType trenchWidth = 3.0;
    const NumericType trenchDepth = 4.0;
    const NumericType taperAngle = 5.0;  // 5 degree taper
    
    // Create a trench domain
    DomainType domain = DomainType::New(gridDelta, xExtent, yExtent);
    
    // Apply the trench geometry
    viennaps::MakeTrench<NumericType, D>(
        domain, 
        trenchWidth, 
        trenchDepth, 
        taperAngle, 
        1.0,   // maskHeight
        2.0,   // maskTaperAngle
        false, // halfTrench
        viennaps::Material::Si, 
        viennaps::Material::SiO2
    ).apply();
    
    // Expand the level sets for better visualization
    for (auto& ls : domain->getLevelSets()) {
        viennals::Expand<NumericType, D>(ls, 2).apply();
    }
    
    // Write the domain to file
    viennaps::Writer<NumericType, D> writer(domain, testFileName);
    writer.apply();
    
    // Verify the file was created
    if (!std::filesystem::exists(testFileName)) {
        std::cerr << "Error: Test file was not created!" << std::endl;
        return 1;
    }
    
    // Read the domain back from file
    viennaps::Reader<NumericType, D> reader(testFileName);
    auto readDomain = reader.apply();
    
    // Verify the domains are equal
    if (domainsEqual(domain, readDomain)) {
        std::cout << "✓ Domain successfully written and read back!" << std::endl;
    } else {
        std::cerr << "✗ Error: Read domain does not match original domain!" << std::endl;
        return 1;
    }
    
    // Print domain information for visual verification
    std::cout << "\nOriginal domain info:" << std::endl;
    std::cout << "Number of level sets: " << domain->getLevelSets().size() << std::endl;
    std::cout << "Grid delta: " << domain->getGridDelta() << std::endl;
    
    std::cout << "\nRead domain info:" << std::endl;
    std::cout << "Number of level sets: " << readDomain->getLevelSets().size() << std::endl;
    std::cout << "Grid delta: " << readDomain->getGridDelta() << std::endl;
    
    
    std::cout << "Test completed successfully!" << std::endl;
    
    // Clean up the test file
    // std::filesystem::remove(testFileName);
    
    return 0;
}
