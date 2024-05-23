#pragma once

#include "psMaterials.hpp"
#include "psSurfacePointValuesToLevelSet.hpp"

#include "cellSet/csDenseCellSet.hpp"

#include <lsBooleanOperation.hpp>
#include <lsDomain.hpp>
#include <lsExpand.hpp>
#include <lsToDiskMesh.hpp>
#include <lsToMesh.hpp>
#include <lsToSurfaceMesh.hpp>
#include <lsVTKWriter.hpp>
#include <lsWriteVisualizationMesh.hpp>
#include <lsWriter.hpp>

#include <vcLogger.hpp>
#include <vcSmartPointer.hpp>

namespace viennaps {

using namespace viennacore;

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
template <class NumericType, int D> class Domain {
public:
  using lsDomainType = SmartPointer<lsDomain<NumericType, D>>;
  using lsDomainsType = SmartPointer<std::vector<lsDomainType>>;
  using csDomainType = SmartPointer<viennacs::DenseCellSet<NumericType, D>>;
  using materialMapType = SmartPointer<MaterialMap>;

  static constexpr char materialIdsLabel[] = "MaterialIds";

private:
  lsDomainsType levelSets_ = nullptr;
  csDomainType cellSet_ = nullptr;
  materialMapType materialMap_ = nullptr;

public:
  // Default constructor.
  Domain() : levelSets_(lsDomainsType::New()) {}

  // Deep copy constructor.
  Domain(SmartPointer<Domain> domain) { deepCopy(domain); }

  // Constructor for domain with a single initial Level-Set.
  Domain(lsDomainType levelSet) : levelSets_(lsDomainsType::New()) {
    levelSets_->push_back(levelSet);
  }

  // Constructor for domain with multiple initial Level-Sets.
  Domain(lsDomainsType levelSets) : levelSets_(levelSets) {}

  // Create a deep copy of all Level-Sets and the Cell-Set from the passed
  // domain.
  void deepCopy(SmartPointer<Domain> domain) {

    // Copy all Level-Sets.
    for (auto &ls : *domain->levelSets_) {
      levelSets_->push_back(lsDomainType::New(ls));
    }

    // Copy material map.
    if (domain->materialMap_) {
      materialMap_ = materialMapType::New();
      for (std::size_t i = 0; i < domain->materialMap_->size(); i++) {
        materialMap_->insertNextMaterial(
            domain->materialMap_->getMaterialAtIdx(i));
      }
    } else {
      materialMap_ = nullptr;
    }

    // Copy Cell-Set.
    if (domain->cellSet_) {
      auto cellSetDepth = domain->getCellSet()->getDepth();
      cellSet_ = csDomainType::New(levelSets_, materialMap_, cellSetDepth);
    } else {
      cellSet_ = nullptr;
    }
  }

  void insertNextLevelSet(lsDomainType levelSet,
                          bool wrapLowerLevelSet = true) {
    if (!levelSets_->empty() && wrapLowerLevelSet) {
      lsBooleanOperation<NumericType, D>(levelSet, levelSets_->back(),
                                         lsBooleanOperationEnum::UNION)
          .apply();
    }
    levelSets_->push_back(levelSet);
    if (materialMap_) {
      Logger::getInstance()
          .addWarning("Inserting non-material specific Level-Set in domain "
                      "with material mapping.")
          .print();
      materialMapCheck();
    }
  }

  void insertNextLevelSetAsMaterial(lsDomainType levelSet,
                                    const Material material,
                                    bool wrapLowerLevelSet = true) {
    if (!levelSets_->empty() && wrapLowerLevelSet) {
      lsBooleanOperation<NumericType, D>(levelSet, levelSets_->back(),
                                         lsBooleanOperationEnum::UNION)
          .apply();
    }
    if (!materialMap_) {
      materialMap_ = materialMapType::New();
    }
    materialMap_->insertNextMaterial(material);
    levelSets_->push_back(levelSet);
    materialMapCheck();
  }

  // Copy the top Level-Set and insert it in the domain (e.g. in order to
  // capture depositing material on top of the surface).
  void duplicateTopLevelSet(const Material material = Material::None) {
    if (levelSets_->empty()) {
      return;
    }

    auto copy = lsDomainType::New(levelSets_->back());
    if (material == Material::None) {
      insertNextLevelSet(copy, false);
    } else {
      insertNextLevelSetAsMaterial(copy, material, false);
    }
  }

  // Remove the top (last inserted) Level-Set.
  void removeTopLevelSet() {
    if (levelSets_->empty()) {
      return;
    }

    levelSets_->pop_back();
    if (materialMap_) {
      auto newMatMap = materialMapType::New();
      for (std::size_t i = 0; i < levelSets_->size(); i++) {
        newMatMap->insertNextMaterial(materialMap_->getMaterialAtIdx(i));
      }
      materialMap_ = newMatMap;
    }
  }

  // Apply a boolean operation with the passed Level-Set to all of the
  // Level-Sets in the domain.
  void applyBooleanOperation(lsDomainType levelSet,
                             lsBooleanOperationEnum operation) {
    if (levelSets_->empty()) {
      return;
    }

    for (auto &layer : *levelSets_) {
      lsBooleanOperation<NumericType, D>(layer, levelSet, operation).apply();
    }
  }

  // Generate the Cell-Set from the Level-Sets in the domain. The Cell-Set can
  // be used to store and track volume data.
  void generateCellSet(const NumericType position, const Material coverMaterial,
                       const bool isAboveSurface = false) {
    if (!cellSet_)
      cellSet_ = csDomainType::New();
    cellSet_->setCellSetPosition(isAboveSurface);
    cellSet_->setCoverMaterial(coverMaterial);
    cellSet_->fromLevelSets(levelSets_, materialMap_, position);
  }

  void setMaterialMap(materialMapType passedMaterialMap) {
    materialMap_ = passedMaterialMap;
    materialMapCheck();
  }

  // Set the material of a specific Level-Set in the domain.
  void setMaterial(unsigned int lsId, const Material material) {
    if (materialMap_) {
      materialMap_ = materialMapType::New();
    }
    materialMap_->setMaterialAtIdx(lsId, material);
    materialMapCheck();
  }

  // Returns a vector with all Level-Sets in the domain.
  auto &getLevelSets() const { return levelSets_; }

  // Returns the material map which contains the specified material for each
  // Level-Set in the domain.
  auto &getMaterialMap() const { return materialMap_; }

  auto &getCellSet() const { return cellSet_; }

  // Returns the underlying HRLE grid of the top Level-Set in the domain.
  auto &getGrid() const { return levelSets_->back()->getGrid(); }

  // Returns the bounding box of the top Level-Set in the domain.
  // [min, max][x, y, z]
  auto getBoundingBox() const {
    std::array<std::array<NumericType, 3>, 2> boundingBox;
    auto mesh = SmartPointer<lsMesh<NumericType>>::New();
    lsToDiskMesh<NumericType, D>(levelSets_->back(), mesh).apply();
    boundingBox[0] = mesh->minimumExtent;
    boundingBox[1] = mesh->maximumExtent;
    return boundingBox;
  }

  void print() const {
    std::cout << "Process Simulation Domain:" << std::endl;
    std::cout << "**************************" << std::endl;
    for (auto &ls : *levelSets_) {
      ls->print();
    }
    std::cout << "**************************" << std::endl;
  }

  // Save the level set as a VTK file.
  void saveLevelSetMesh(std::string fileName, int width = 1) {
    for (int i = 0; i < levelSets_->size(); i++) {
      auto mesh = SmartPointer<lsMesh<NumericType>>::New();
      lsExpand<NumericType, D>(levelSets_->at(i), width).apply();
      lsToMesh<NumericType, D>(levelSets_->at(i), mesh).apply();
      lsVTKWriter<NumericType>(mesh,
                               fileName + "_layer" + std::to_string(i) + ".vtp")
          .apply();
    }
  }

  // Print the top Level-Set (surface) in a VTK file format (recommended: .vtp).
  void saveSurfaceMesh(std::string fileName, bool addMaterialIds = true) {

    auto mesh = SmartPointer<lsMesh<NumericType>>::New();

    if (addMaterialIds) {
      lsToDiskMesh<NumericType, D> meshConverter;
      meshConverter.setMesh(mesh);
      if (materialMap_)
        meshConverter.setMaterialMap(materialMap_->getMaterialMap());
      for (const auto ls : *levelSets_) {
        meshConverter.insertNextLevelSet(ls);
      }
      meshConverter.apply();

      psSurfacePointValuesToLevelSet<NumericType, D>(levelSets_->back(), mesh,
                                                     {"MaterialIds"})
          .apply();
    }

    lsToSurfaceMesh<NumericType, D>(levelSets_->back(), mesh).apply();
    lsVTKWriter<NumericType>(mesh, fileName).apply();
  }

  // Save the domain as a volume mesh
  void saveVolumeMesh(std::string fileName) const {
    lsWriteVisualizationMesh<NumericType, D> visMesh;
    visMesh.setFileName(fileName);
    for (auto &ls : *levelSets_) {
      visMesh.insertNextLevelSet(ls);
    }
    if (materialMap_)
      visMesh.setMaterialMap(materialMap_->getMaterialMap());
    visMesh.apply();
  }

  // Write the all Level-Sets in the domain to individual files. The file name
  // serves as the prefix for the individual files and is append by
  // "_layerX.lvst", where X is the number of the Level-Set in the domain.
  void saveLevelSets(std::string fileName) const {
    for (int i = 0; i < levelSets_->size(); i++) {
      lsWriter<NumericType, D>(
          levelSets_->at(i), fileName + "_layer" + std::to_string(i) + ".lvst")
          .apply();
    }
  }

  void clear() {
    levelSets_ = lsDomainsType::New();
    if (cellSet_)
      cellSet_ = csDomainType::New();
    if (materialMap_)
      materialMap_ = materialMapType::New();
  }

private:
  void materialMapCheck() const {
    if (!materialMap_)
      return;

    if (materialMap_->size() != levelSets_->size()) {
      Logger::getInstance()
          .addWarning("Size mismatch in material map and number of Level-Sets "
                      "in domain.")
          .print();
    }
  }
};

} // namespace viennaps
