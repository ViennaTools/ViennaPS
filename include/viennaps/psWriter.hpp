#pragma once

#include "psDomain.hpp"
#include "psPreCompileMacros.hpp"

#include <fstream>
#include <string>
#include <utility>

#include <vcLogger.hpp>
#include <vcSmartPointer.hpp>

namespace viennaps {

using namespace viennacore;

/**
 * @brief Writer class for serializing and writing a psDomain to a file
 *
 * This class handles serializing a Process Simulation Domain (psDomain) to a
 * binary file. The file format (.vpsd - ViennaPS Domain) contains all
 * levelSets, cell data, material mappings and domain setup information.
 */
template <class NumericType, int D> class Writer {
private:
  SmartPointer<Domain<NumericType, D>> domain = nullptr;
  std::string fileName;

public:
  Writer() = default;

  Writer(SmartPointer<Domain<NumericType, D>> passedDomain)
      : domain(passedDomain) {}

  Writer(SmartPointer<Domain<NumericType, D>> passedDomain,
         std::string passedFileName)
      : domain(passedDomain), fileName(std::move(passedFileName)) {}

  void setDomain(SmartPointer<Domain<NumericType, D>> passedDomain) {
    domain = passedDomain;
  }

  /// set file name for file to write
  void setFileName(std::string passedFileName) {
    fileName = std::move(passedFileName);
  }

  void apply() {
    // check domain
    if (domain == nullptr) {
      Logger::getInstance()
          .addWarning("No domain was passed to Writer. Not writing.")
          .print();
      return;
    }

    // check filename
    if (fileName.empty()) {
      Logger::getInstance()
          .addWarning("No file name specified for Writer. Not writing.")
          .print();
      return;
    }

    if (fileName.find(".vpsd") != fileName.length() - 5) {
      Logger::getInstance()
          .addWarning("File name does not end in '.vpsd', appending it.")
          .print();
      fileName.append(".vpsd");
    }

    // Open file for writing and save serialized domain
    std::ofstream fout(fileName, std::ios::binary);

    // Write header identifier
    fout << "psDomain";

    // Write version number (starting with 0)
    char formatVersion = 0;
    fout.write(&formatVersion, 1);

    // Write domain setup
    auto &setup = domain->getSetup();
    fout.write(reinterpret_cast<const char *>(&setup), sizeof(setup));

    // Write number of level sets
    auto &levelSets = domain->getLevelSets();
    uint32_t numLevelSets = levelSets.size();
    fout.write(reinterpret_cast<const char *>(&numLevelSets), sizeof(uint32_t));

    // Write each level set
    for (auto &ls : levelSets) {
      ls->serialize(fout);
    }

    // Write material map if it exists
    auto &materialMap = domain->getMaterialMap();
    char hasMaterialMap = (materialMap == nullptr) ? 0 : 1;
    fout.write(&hasMaterialMap, 1);

    if (hasMaterialMap) {
      // Write number of materials
      uint32_t numMaterials = materialMap->size();
      fout.write(reinterpret_cast<const char *>(&numMaterials),
                 sizeof(uint32_t));

      // Write each material ID
      for (size_t i = 0; i < numMaterials; i++) {
        int materialId = static_cast<int>(materialMap->getMaterialAtIdx(i));
        fout.write(reinterpret_cast<const char *>(&materialId), sizeof(int));
      }
    }

    // Write cell set if it exists
    auto &cellSet = domain->getCellSet();
    char hasCellSet = (cellSet == nullptr) ? 0 : 1;
    fout.write(&hasCellSet, 1);

    if (hasCellSet) {
      // Serialize cell set
      // This would require implementing serialization for the cell set
      // For now, just include a placeholder for future implementation
      Logger::getInstance()
          .addWarning("CellSet serialization not yet implemented in psWriter.")
          .print();
    }

    fout.close();
  }
};

PS_PRECOMPILE_PRECISION_DIMENSION(Writer)

} // namespace viennaps
