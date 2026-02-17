#pragma once

#include "psDomain.hpp"
#include "psPreCompileMacros.hpp"

#include <cstring>
#include <fstream>
#include <string>
#include <utility>

#include <lsDomain.hpp>
#include <lsReader.hpp>

#include <vcLogger.hpp>
#include <vcSmartPointer.hpp>

namespace viennaps {

using namespace viennacore;

/// @brief Reader class for deserializing a Domain from a file
///
/// This class handles reading a Process Simulation Domain (Domain) from a
/// binary file previously created with psWriter.
VIENNAPS_TEMPLATE_ND(NumericType, D) class Reader {
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

  void apply() {
    if (!domain) {
      VIENNACORE_LOG_ERROR("No domain was passed to Reader.");
      return;
    }

    // check filename
    if (fileName.empty()) {
      VIENNACORE_LOG_ERROR("No file name specified for Reader. Not reading.");
      return;
    }

    // Open file for reading
    std::ifstream fin(fileName, std::ios::binary);
    if (!fin.good()) {
      VIENNACORE_LOG_ERROR("Could not open file: " + fileName);
      return;
    }

    // Check identifier
    char identifier[8];
    fin.read(identifier, 8);
    if (std::memcmp(identifier, "psDomain", 8) != 0) {
      VIENNACORE_LOG_ERROR(
          "Reading domain from stream failed. Header could not be found.");
      return;
    }

    // Check format version
    char formatVersion;
    fin.read(&formatVersion, 1);
    if (formatVersion > 0) { // Update this when version changes
      VIENNACORE_LOG_ERROR("Reading domain of version " +
                           std::to_string(formatVersion) +
                           " with reader of version 0 failed.");
      return;
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
    std::vector<SmartPointer<viennals::Domain<NumericType, D>>> levelSets;
    for (uint32_t i = 0; i < numLevelSets; i++) {
      auto ls = viennals::Domain<NumericType, D>::New();
      ls->deserialize(fin);
      levelSets.push_back(ls);
    }
    assert(levelSets.size() == numLevelSets);

    // Read material map if it exists
    char hasMaterialMap;
    fin.read(&hasMaterialMap, 1);

    if (hasMaterialMap) {
      // Read number of materials
      uint32_t numMaterials;
      fin.read(reinterpret_cast<char *>(&numMaterials), sizeof(uint32_t));
      assert(numMaterials == numLevelSets);

      // Read each material ID and insert corresponding level set
      for (uint32_t i = 0; i < numMaterials; i++) {
        int materialId;
        fin.read(reinterpret_cast<char *>(&materialId), sizeof(int));
        domain->insertNextLevelSetAsMaterial(
            levelSets[i], static_cast<Material>(materialId), false);
      }

    } else {
      VIENNACORE_LOG_ERROR("No material map found in the file.");
    }

    // Check if cell set exists
    char hasCellSet;
    fin.read(&hasCellSet, 1);

    if (hasCellSet) {
      // Deserialize cell set
      // This would require implementing deserialization for the cell set
      // For now, just include a placeholder for future implementation
      VIENNACORE_LOG_WARNING(
          "CellSet deserialization not yet implemented in psReader.");
    }

    fin.close();
  }
};

PS_PRECOMPILE_PRECISION_DIMENSION(Reader)

} // namespace viennaps
