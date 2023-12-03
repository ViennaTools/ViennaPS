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
  It contains level sets for the accurate surface representation
  and a cell-based structure for the storage of volume information.
  These structures are used depending on the process applied to the material.
  Processes may use one of either structures or both.
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
  psDomain() : levelSets(lsDomainsType::New()) {}

  psDomain(psSmartPointer<psDomain> passedDomain) { deepCopy(passedDomain); }

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

  // copy the top LS and insert it in the domain (used to capture depositing
  // material)
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

  // remove the top LS
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

  // Boolean Operation of all level sets in the domain with the specified LS
  void applyBooleanOperation(lsDomainType levelSet,
                             lsBooleanOperationEnum operation) {
    if (levelSets->empty()) {
      return;
    }

    for (auto layer : *levelSets) {
      lsBooleanOperation<NumericType, D>(layer, levelSet, operation).apply();
    }
  }

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

  void setMaterial(unsigned int lsId, const psMaterial material) {
    if (materialMap) {
      materialMap = materialMapType::New();
    }
    materialMap->setMaterialAtIdx(lsId, material);
    materialMapCheck();
  }

  auto &getLevelSets() const { return levelSets; }

  auto &getMaterialMap() const { return materialMap; }

  auto &getCellSet() const { return cellSet; }

  auto &getGrid() const { return levelSets->back()->getGrid(); }

  void print() const {
    std::cout << "Process Simulation Domain:" << std::endl;
    std::cout << "**************************" << std::endl;
    for (auto &ls : *levelSets) {
      ls->print();
    }
    std::cout << "**************************" << std::endl;
  }

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
