#pragma once

#include <lsBooleanOperation.hpp>
#include <lsDomain.hpp>
#include <lsMakeGeometry.hpp>
#include <lsToDiskMesh.hpp>
#include <lsToSurfaceMesh.hpp>
#include <lsWriter.hpp>

#include <csDenseCellSet.hpp>

#include <psMaterials.hpp>
#include <psSmartPointer.hpp>
#include <psSurfacePointValuesToLevelSet.hpp>
#include <psVTKWriter.hpp>

/**
  This class represents all materials in the simulation domain.
  It contains Level-Sets for an accurate surface representation
  and a cell-based structure for the storage of volume information.
  These structures are used depending on the process applied to the material.
  Processes may use one of either structure or both.

  Level-Sets in the domain automatically wrap all lower domains when inserted.
  If specified, each Level-Set is assigned a specific material,
  which can be used in a process to implement material specific rates or
  similar.
*/
template <class NumericType = float, int D = 3> class psDomain {
public:
  typedef psSmartPointer<lsDomain<NumericType, D>> lsDomainType;
  typedef psSmartPointer<std::vector<lsDomainType>> lsDomainsType;
  typedef psSmartPointer<csDenseCellSet<NumericType, D>> csDomainType;
  typedef psSmartPointer<psMaterialMap> materialMapType;

  static constexpr char materialIdsLabel[] = "MaterialIds";

private:
  lsDomainsType levelSets = nullptr;
  csDomainType cellSet = nullptr;
  materialMapType materialMap = nullptr;
  NumericType cellSetDepth = 0.;

public:
  // Default constructor.
  psDomain() : levelSets(lsDomainsType::New()) {}

  // Deep copy constructor.
  psDomain(psSmartPointer<psDomain> passedDomain) { deepCopy(passedDomain); }

  // Constructor for domain with a single initial Level-Set.
  psDomain(lsDomainType passedLevelSet, bool generateCellSet = false,
           const NumericType passedCellSetDepth = 0.,
           const bool passedCellSetPosition = false)
      : levelSets(lsDomainsType::New()), cellSetDepth(passedCellSetDepth) {
    levelSets->push_back(passedLevelSet);
    // generate CellSet
    if (generateCellSet) {
      cellSet = csDomainType::New(levelSets, materialMap, cellSetDepth,
                                  passedCellSetPosition);
    }
  }

  // Constructor for domain with multiple initial Level-Sets.
  psDomain(lsDomainsType passedLevelSets, bool generateCellSet = false,
           const NumericType passedCellSetDepth = 0.,
           const bool passedCellSetPosition = false)
      : levelSets(passedLevelSets), cellSetDepth(passedCellSetDepth) {
    // generate CellSet
    if (generateCellSet) {
      cellSet = csDomainType::New(levelSets, materialMap, cellSetDepth,
                                  passedCellSetPosition);
    }
  }

  // Create a deep copy of all Level-Set and the Cell-Set in the passed domain.
  void deepCopy(psSmartPointer<psDomain> passedDomain) {
    unsigned numLevelSets = passedDomain->levelSets->size();
    for (unsigned i = 0; i < numLevelSets; ++i) {
      levelSets->push_back(lsSmartPointer<lsDomain<NumericType, D>>::New(
          passedDomain->levelSets->at(i)));
    }
    if (passedDomain->materialMap) {
      materialMap = materialMapType::New();
      for (std::size_t i = 0; i < passedDomain->materialMap->size(); i++) {
        materialMap->insertNextMaterial(
            passedDomain->materialMap->getMaterialAtIdx(i));
      }
    } else {
      materialMap = nullptr;
    }
    if (passedDomain->cellSet) {
      cellSetDepth = passedDomain->cellSetDepth;
      cellSet = csDomainType::New(levelSets, materialMap, cellSetDepth);
    } else {
      cellSet = nullptr;
    }
  }

  void insertNextLevelSet(lsDomainType passedLevelSet,
                          bool wrapLowerLevelSet = true) {
    if (!levelSets->empty() && wrapLowerLevelSet) {
      lsBooleanOperation<NumericType, D>(passedLevelSet, levelSets->back(),
                                         lsBooleanOperationEnum::UNION)
          .apply();
    }
    levelSets->push_back(passedLevelSet);
    if (materialMap) {
      psLogger::getInstance()
          .addWarning("Inserting non-material specific Level-Set in domain "
                      "with material mapping.")
          .print();
      materialMapCheck();
    }
  }

  void insertNextLevelSetAsMaterial(lsDomainType passedLevelSet,
                                    const psMaterial material,
                                    bool wrapLowerLevelSet = true) {
    if (!levelSets->empty() && wrapLowerLevelSet) {
      lsBooleanOperation<NumericType, D>(passedLevelSet, levelSets->back(),
                                         lsBooleanOperationEnum::UNION)
          .apply();
    }
    if (!materialMap) {
      materialMap = materialMapType::New();
    }
    materialMap->insertNextMaterial(material);
    levelSets->push_back(passedLevelSet);
    materialMapCheck();
  }

  // Copy the top Level-Set and insert it in the domain (e.g. in order to
  // capture depositing material on top of the surface).
  void duplicateTopLevelSet(const psMaterial material = psMaterial::None) {
    if (levelSets->empty()) {
      return;
    }

    auto copy = lsDomainType::New(levelSets->back());
    if (material == psMaterial::None) {
      insertNextLevelSet(copy, false);
    } else {
      insertNextLevelSetAsMaterial(copy, material, false);
    }
  }

  // Remove the top (last inserted) Level-Set.
  void removeTopLevelSet() {
    if (levelSets->empty()) {
      return;
    }

    levelSets->pop_back();
    if (materialMap) {
      auto newMatMap = materialMapType::New();
      for (std::size_t i = 0; i < levelSets->size(); i++) {
        newMatMap->insertNextMaterial(materialMap->getMaterialAtIdx(i));
      }
      materialMap = newMatMap;
    }
  }

  // Apply a boolean operation with the passed Level-Set to all of the
  // Level-Sets in the domain.
  void applyBooleanOperation(lsDomainType levelSet,
                             lsBooleanOperationEnum operation) {
    if (levelSets->empty()) {
      return;
    }

    for (auto layer : *levelSets) {
      lsBooleanOperation<NumericType, D>(layer, levelSet, operation).apply();
    }
  }

  // Generate the Cell-Set from the Level-Sets in the domain. The Cell-Set can
  // be used to store and track volume data.
  void generateCellSet(const NumericType depth = 0.,
                       const bool passedCellSetPosition = false) {
    cellSetDepth = depth;
    if (!cellSet)
      cellSet = csDomainType::New();
    cellSet->setCellSetPosition(passedCellSetPosition);
    cellSet->fromLevelSets(levelSets, materialMap, cellSetDepth);
  }

  void setMaterialMap(materialMapType passedMaterialMap) {
    materialMap = passedMaterialMap;
    materialMapCheck();
  }

  // Set the material of a specific Level-Set in the domain.
  void setMaterial(unsigned int lsId, const psMaterial material) {
    if (materialMap) {
      materialMap = materialMapType::New();
    }
    materialMap->setMaterialAtIdx(lsId, material);
    materialMapCheck();
  }

  // Returns a vector with all Level-Sets in the domain.
  auto &getLevelSets() const { return levelSets; }

  // Returns the material map which contains the specified material for each
  // Level-Set in the domain.
  auto &getMaterialMap() const { return materialMap; }

  auto &getCellSet() const { return cellSet; }

  // Returns the underlying HRLE grid of the top Level-Set in the domain.
  auto &getGrid() const { return levelSets->back()->getGrid(); }

  void print() const {
    std::cout << "Process Simulation Domain:" << std::endl;
    std::cout << "**************************" << std::endl;
    for (auto &ls : *levelSets) {
      ls->print();
    }
    std::cout << "**************************" << std::endl;
  }

  // Print the top Level-Set (surface) in a VTK file format (recommended: .vtp).
  void printSurface(std::string name, bool addMaterialIds = true) {

    auto mesh = psSmartPointer<lsMesh<NumericType>>::New();

    if (addMaterialIds) {
      lsToDiskMesh<NumericType, D> meshConverter;
      meshConverter.setMesh(mesh);
      if (materialMap)
        meshConverter.setMaterialMap(materialMap->getMaterialMap());
      for (const auto ls : *levelSets) {
        meshConverter.insertNextLevelSet(ls);
      }
      meshConverter.apply();

      psSurfacePointValuesToLevelSet<NumericType, D>(levelSets->back(), mesh,
                                                     {"MaterialIds"})
          .apply();
    }

    lsToSurfaceMesh<NumericType, D>(levelSets->back(), mesh).apply();
    psVTKWriter<NumericType>(mesh, name).apply();
  }

  // Write the all Level-Sets in the domain to individual files. The file name
  // serves as the prefix for the individual files and is append by
  // "_layerX.lvst", where X is the number of the Level-Set in the domain.
  void writeLevelSets(std::string fileName) const {
    for (int i = 0; i < levelSets->size(); i++) {
      lsWriter<NumericType, D>(
          levelSets->at(i), fileName + "_layer" + std::to_string(i) + ".lvst")
          .apply();
    }
  }

  void clear() {
    levelSets = lsDomainsType::New();
    if (cellSet)
      cellSet = csDomainType::New();
    if (materialMap)
      materialMap = materialMapType::New();
  }

private:
  void materialMapCheck() const {
    if (!materialMap)
      return;

    if (materialMap->size() != levelSets->size()) {
      psLogger::getInstance()
          .addWarning("Size mismatch in material map and number of Level-Sets "
                      "in domain.")
          .print();
    }
  }
};
