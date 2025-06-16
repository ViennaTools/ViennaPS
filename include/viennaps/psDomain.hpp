#pragma once

#include "psDomainSetup.hpp"
#include "psMaterials.hpp"
#include "psSurfacePointValuesToLevelSet.hpp"

#include <lsBooleanOperation.hpp>
#include <lsDomain.hpp>
#include <lsExpand.hpp>
#include <lsToDiskMesh.hpp>
#include <lsToMesh.hpp>
#include <lsToSurfaceMesh.hpp>
#include <lsVTKWriter.hpp>
#include <lsWriteVisualizationMesh.hpp>
#include <lsWriter.hpp>

#include <csDenseCellSet.hpp>

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
  using lsDomainType = SmartPointer<viennals::Domain<NumericType, D>>;
  using lsDomainsType = std::vector<lsDomainType>;
  using csDomainType = SmartPointer<viennacs::DenseCellSet<NumericType, D>>;
  using materialMapType = SmartPointer<MaterialMap>;
  using Setup = DomainSetup<NumericType, D>;

  static constexpr char materialIdsLabel[] = "MaterialIds";

private:
  Setup setup_;
  lsDomainsType levelSets_;
  csDomainType cellSet_ = nullptr;
  materialMapType materialMap_ = nullptr;

public:
  // Default constructor.
  Domain() = default;

  // Deep copy constructor.
  explicit Domain(SmartPointer<Domain> domain) { deepCopy(domain); }

  // Constructor for domain with a single initial Level-Set.
  explicit Domain(lsDomainType levelSet) {
    setup_.init(levelSet->getGrid());
    levelSets_.push_back(levelSet);
  }

  // Constructor for domain with multiple initial Level-Sets.
  explicit Domain(lsDomainsType levelSets) : levelSets_(levelSets) {
    setup_.init(levelSets.back()->getGrid());
  }

  // Sets up domain in with primary direction y in 2D and z in 3D
  Domain(NumericType gridDelta, NumericType xExtent,
         BoundaryType boundary = BoundaryType::REFLECTIVE_BOUNDARY)
      : setup_(gridDelta, xExtent, 0.0, boundary) {
    static_assert(D == 2, "Domain setup only valid for 2D.");
  }

  // Sets up domain in with primary direction y in 2D and z in 3D
  // In 2D yExtent is ignored.
  Domain(NumericType gridDelta, NumericType xExtent, NumericType yExtent = 0.0,
         BoundaryType boundary = BoundaryType::REFLECTIVE_BOUNDARY)
      : setup_(gridDelta, xExtent, yExtent, boundary) {}

  // Convenience function to create a new domain.
  template <class... Args> static auto New(Args &&...args) {
    return SmartPointer<Domain>::New(std::forward<Args>(args)...);
  }

  explicit Domain(const Setup &setup) : setup_(setup) {}

  void setup(const Setup &setup) { setup_ = setup; }

  void setup(NumericType gridDelta, NumericType xExtent,
             NumericType yExtent = 0,
             BoundaryType boundary = BoundaryType::REFLECTIVE_BOUNDARY) {
    setup_ = Setup(gridDelta, xExtent, yExtent, boundary);
  }

  // Create a deep copy of all Level-Sets and the Cell-Set from the passed
  // domain.
  void deepCopy(SmartPointer<Domain> domain) {

    clear();
    setup_ = domain->setup_;

    // Copy all Level-Sets.
    for (auto &ls : domain->levelSets_) {
      levelSets_.push_back(lsDomainType::New(ls));
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
      cellSet_ = csDomainType::New(levelSets_, materialMap_->getMaterialMap(),
                                   cellSetDepth);
    } else {
      cellSet_ = nullptr;
    }
  }

  // Will be deprecated in the future. Please use insertNextLevelSetAsMaterial
  // instead.
  void insertNextLevelSet(lsDomainType levelSet,
                          bool wrapLowerLevelSet = true) {
    if (levelSets_.empty()) {
      setup_.init(levelSet->getGrid());
    }
    if (!levelSets_.empty() && wrapLowerLevelSet) {
      viennals::BooleanOperation<NumericType, D>(
          levelSet, levelSets_.back(), viennals::BooleanOperationEnum::UNION)
          .apply();
    }
    levelSets_.push_back(levelSet);
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
    if (levelSets_.empty()) {
      setup_.init(levelSet->getGrid());
    }
    if (std::abs(levelSet->getGrid().getGridDelta() - setup_.gridDelta()) >
        1e-6) {
      Logger::getInstance()
          .addError("Grid delta of Level-Set does not match domain grid "
                    "delta.")
          .print();
    }
    if (!levelSets_.empty() && wrapLowerLevelSet) {
      viennals::BooleanOperation<NumericType, D>(
          levelSet, levelSets_.back(), viennals::BooleanOperationEnum::UNION)
          .apply();
    }
    if (!materialMap_) {
      materialMap_ = materialMapType::New();
    }
    materialMap_->insertNextMaterial(material);
    levelSets_.push_back(levelSet);
    materialMapCheck();
  }

  // Copy the top Level-Set and insert it in the domain (e.g. in order to
  // capture depositing material on top of the surface).
  void duplicateTopLevelSet(const Material material) {
    if (levelSets_.empty()) {
      Logger::getInstance()
          .addWarning("Trying to duplicate non-existing Level-Set in domain.")
          .print();
      return;
    }

    auto copy = lsDomainType::New(levelSets_.back());
    insertNextLevelSetAsMaterial(copy, material, false);
  }

  // Remove the top (last inserted) Level-Set.
  void removeTopLevelSet() {
    if (levelSets_.empty()) {
      return;
    }

    levelSets_.pop_back();
    if (materialMap_) {
      auto newMatMap = materialMapType::New();
      for (std::size_t i = 0; i < levelSets_.size(); i++) {
        newMatMap->insertNextMaterial(materialMap_->getMaterialAtIdx(i));
      }
      materialMap_ = newMatMap;
    }
  }

  // Apply a boolean operation with the passed Level-Set to all
  // Level-Sets in the domain.
  void applyBooleanOperation(lsDomainType levelSet,
                             viennals::BooleanOperationEnum operation) {
    if (levelSets_.empty()) {
      return;
    }

    for (auto &layer : levelSets_) {
      viennals::BooleanOperation<NumericType, D>(layer, levelSet, operation)
          .apply();
    }
  }

  void removeLevelSet(unsigned int idx, bool removeWrapped = true) {
    if (idx >= levelSets_.size()) {
      Logger::getInstance()
          .addWarning("Trying to remove non-existing Level-Set from domain.")
          .print();
      return;
    }

    if (materialMap_) {
      auto newMatMap = materialMapType::New();
      for (std::size_t i = 0; i < levelSets_.size(); i++) {
        if (i == idx)
          continue;

        newMatMap->insertNextMaterial(materialMap_->getMaterialAtIdx(i));
      }
      materialMap_ = newMatMap;
    }

    if (removeWrapped) {
      auto remove = levelSets_.at(idx);

      for (int i = idx - 1; i >= 0; i--) {
        viennals::BooleanOperation<NumericType, D>(
            remove, levelSets_.at(i),
            viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT)
            .apply();
      }

      for (int i = idx + 1; i < levelSets_.size(); i++) {
        viennals::BooleanOperation<NumericType, D>(
            levelSets_.at(i), remove,
            viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT)
            .apply();
      }
    }

    levelSets_.erase(levelSets_.begin() + idx);
    materialMapCheck();
  }

  void removeMaterial(const Material material) {
    if (!materialMap_) {
      return;
    }

    for (int i = 0; i < materialMap_->size(); i++) {
      if (materialMap_->getMaterialAtIdx(i) == material) {
        removeLevelSet(i);
        i -= 1;
      }
    }
  }

  // Generate the Cell-Set from the Level-Sets in the domain. The Cell-Set can
  // be used to store and track volume data.
  void generateCellSet(const NumericType position, const Material coverMaterial,
                       const bool isAboveSurface = false) {
    if (!cellSet_)
      cellSet_ = csDomainType::New();
    cellSet_->setCellSetPosition(isAboveSurface);
    cellSet_->setCoverMaterial(static_cast<int>(coverMaterial));
    cellSet_->fromLevelSets(
        levelSets_, materialMap_ ? materialMap_->getMaterialMap() : nullptr,
        position);
  }

  void setMaterialMap(const materialMapType &passedMaterialMap) {
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

  // Returns the top Level-Set (surface) in the domain.
  auto &getSurface() const { return levelSets_.back(); }

  // Returns a vector with all Level-Sets in the domain.
  auto &getLevelSets() const { return levelSets_; }

  // Returns the material map which contains the specified material for each
  // Level-Set in the domain.
  auto &getMaterialMap() const { return materialMap_; }

  auto &getCellSet() const { return cellSet_; }

  // Returns the underlying HRLE grid of the top Level-Set in the domain.
  auto &getGrid() const { return setup_.grid(); }

  auto getGridDelta() const { return setup_.gridDelta(); }

  auto &getSetup() { return setup_; }

  // Returns the bounding box of the top Level-Set in the domain.
  // [min, max][x, y, z]
  auto getBoundingBox() const {
    std::array<Vec3D<NumericType>, 2> boundingBox;
    auto mesh = viennals::Mesh<NumericType>::New();
    viennals::ToDiskMesh<NumericType, D>(levelSets_.back(), mesh).apply();
    boundingBox[0] = mesh->minimumExtent;
    boundingBox[1] = mesh->maximumExtent;
    return boundingBox;
  }

  // Returns the boundary conditions of the domain.
  auto getBoundaryConditions() const {
    return levelSets_.back()->getGrid().getBoundaryConditions();
  }

  void print() const {
    std::cout << "Process Simulation Domain:" << std::endl;
    std::cout << "**************************" << std::endl;
    for (auto &ls : levelSets_) {
      ls->print();
    }
    std::cout << "**************************" << std::endl;
  }

  // Save the level set as a VTK file.
  void saveLevelSetMesh(const std::string &fileName, int width = 1) {
    for (int i = 0; i < levelSets_.size(); i++) {
      auto mesh = viennals::Mesh<NumericType>::New();
      viennals::Expand<NumericType, D>(levelSets_.at(i), width).apply();
      viennals::ToMesh<NumericType, D>(levelSets_.at(i), mesh).apply();
      viennals::VTKWriter<NumericType>(mesh, fileName + "_layer" +
                                                 std::to_string(i) + ".vtp")
          .apply();
    }
  }

  // Print the top Level-Set (surface) in a VTK file format (recommended: .vtp).
  void saveSurfaceMesh(std::string fileName, bool addMaterialIds = true) {

    auto mesh = viennals::Mesh<NumericType>::New();

    if (addMaterialIds) {
      viennals::ToDiskMesh<NumericType, D> meshConverter;
      meshConverter.setMesh(mesh);
      if (materialMap_)
        meshConverter.setMaterialMap(materialMap_->getMaterialMap());
      for (const auto ls : levelSets_) {
        meshConverter.insertNextLevelSet(ls);
      }
      meshConverter.apply();

      SurfacePointValuesToLevelSet<NumericType, D>(levelSets_.back(), mesh,
                                                   {"MaterialIds"})
          .apply();
    }

    viennals::ToSurfaceMesh<NumericType, D>(levelSets_.back(), mesh).apply();
    viennals::VTKWriter<NumericType>(mesh, fileName).apply();
  }

  // Save the domain as a volume mesh
  void saveVolumeMesh(std::string fileName,
                      double wrappingLayerEpsilon = 1e-2) const {
    viennals::WriteVisualizationMesh<NumericType, D> visMesh;
    visMesh.setFileName(fileName);
    visMesh.setWrappingLayerEpsilon(wrappingLayerEpsilon);
    for (auto &ls : levelSets_) {
      visMesh.insertNextLevelSet(ls);
    }
    if (materialMap_)
      visMesh.setMaterialMap(materialMap_->getMaterialMap());
    visMesh.apply();
  }

  void saveHullMesh(std::string fileName,
                    double wrappingLayerEpsilon = 1e-2) const {
    viennals::WriteVisualizationMesh<NumericType, D> visMesh;
    visMesh.setFileName(fileName);
    visMesh.setWrappingLayerEpsilon(wrappingLayerEpsilon);
    visMesh.setExtractHullMesh(true);
    visMesh.setExtractVolumeMesh(false);
    for (auto &ls : levelSets_) {
      visMesh.insertNextLevelSet(ls);
    }
    if (materialMap_)
      visMesh.setMaterialMap(materialMap_->getMaterialMap());
    visMesh.apply();
  }

  // Write the all Level-Sets in the domain to individual files. The file name
  // serves as the prefix for the individual files and is appended by
  // "_layerX.lvst", where X is the number of the Level-Set in the domain.
  void saveLevelSets(const std::string &fileName) const {
    for (int i = 0; i < levelSets_.size(); i++) {
      viennals::Writer<NumericType, D>(
          levelSets_.at(i), fileName + "_layer" + std::to_string(i) + ".lvst")
          .apply();
    }
  }

  void clear() {
    levelSets_.clear();
    if (cellSet_)
      cellSet_ = csDomainType::New();
    if (materialMap_)
      materialMap_ = materialMapType::New();
  }

private:
  void materialMapCheck() const {
    if (!materialMap_)
      return;

    if (materialMap_->size() != levelSets_.size()) {
      Logger::getInstance()
          .addWarning("Size mismatch in material map and number of Level-Sets "
                      "in domain.")
          .print();
    }
  }
};

} // namespace viennaps
