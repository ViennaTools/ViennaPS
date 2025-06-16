#pragma once

#include "psDomain.hpp"

#include <fstream>
#include <string>
#include <utility>

#include <lsDomain.hpp>
#include <lsReader.hpp>

#include <vcLogger.hpp>
#include <vcSmartPointer.hpp>

namespace viennaps {

using namespace viennacore;

/**
 * @brief Reader class for deserializing a psDomain from a file
 *
 * This class handles reading a Process Simulation Domain (psDomain) from a
 * binary file previously created with psWriter.
 */
template <class NumericType, int D> class Reader {
private:
  SmartPointer<Domain<NumericType, D>> domain = nullptr;
  std::string fileName;

public:
  Reader() = default;

  Reader(std::string passedFileName) : fileName(std::move(passedFileName)) {}

  Reader(SmartPointer<Domain<NumericType, D>> passedDomain,
         std::string passedFileName)
      : domain(passedDomain), fileName(std::move(passedFileName)) {}

  void setDomain(SmartPointer<Domain<NumericType, D>> passedDomain) {
    domain = passedDomain;
  }

  /// set file name to read from
  void setFileName(std::string passedFileName) {
    fileName = std::move(passedFileName);
  }

  SmartPointer<Domain<NumericType, D>> apply() {
    // Create new domain if none was provided
    if (domain == nullptr) {
      domain = SmartPointer<Domain<NumericType, D>>::New();
    }

    // check filename
    if (fileName.empty()) {
      Logger::getInstance()
          .addWarning("No file name specified for Reader. Not reading.")
          .print();
      return domain;
    }

    // Open file for reading
    std::ifstream fin(fileName, std::ios::binary);
    if (!fin.good()) {
      Logger::getInstance()
          .addWarning("Could not open file: " + fileName)
          .print();
      return domain;
    }

    // Check identifier
    char identifier[8];
    fin.read(identifier, 8);
    if (std::string(identifier).compare(0, 8, "psDomain")) {
      Logger::getInstance()
          .addWarning(
              "Reading domain from stream failed. Header could not be found.")
          .print();
      return domain;
    }

    // Check format version
    char formatVersion;
    fin.read(&formatVersion, 1);
    if (formatVersion > 0) { // Update this when version changes
      Logger::getInstance()
          .addWarning("Reading domain of version " +
                      std::to_string(formatVersion) +
                      " with reader of version 0 failed.")
          .print();
      return domain;
    }

    // Clear existing domain data
    domain->clear();

    // Read domain setup
    typename Domain<NumericType, D>::Setup setup;
    fin.read(reinterpret_cast<char *>(&setup), sizeof(setup));
    domain->setup(setup);

    // Read number of level sets
    uint32_t numLevelSets;
    fin.read(reinterpret_cast<char *>(&numLevelSets), sizeof(uint32_t));

    // Read each level set
    for (uint32_t i = 0; i < numLevelSets; i++) {
      auto ls = viennals::Domain<NumericType, D>::New();
      ls->deserialize(fin);
      domain->insertNextLevelSet(ls, false); // Don't wrap lower level sets
    }

    // Read material map if it exists
    char hasMaterialMap;
    fin.read(&hasMaterialMap, 1);

    if (hasMaterialMap) {
      // Read number of materials
      uint32_t numMaterials;
      fin.read(reinterpret_cast<char *>(&numMaterials), sizeof(uint32_t));

      // Create new material map
      auto materialMap = SmartPointer<MaterialMap>::New();

      // Read each material ID
      for (uint32_t i = 0; i < numMaterials; i++) {
        int materialId;
        fin.read(reinterpret_cast<char *>(&materialId), sizeof(int));
        materialMap->insertNextMaterial(static_cast<Material>(materialId));
      }

      domain->setMaterialMap(materialMap);
    }

    // Check if cell set exists
    char hasCellSet;
    fin.read(&hasCellSet, 1);

    if (hasCellSet) {
      // Deserialize cell set
      // This would require implementing deserialization for the cell set
      // For now, just include a placeholder for future implementation
      Logger::getInstance()
          .addWarning(
              "CellSet deserialization not yet implemented in psReader.")
          .print();
    }

    fin.close();
    return domain;
  }
};

} // namespace viennaps
