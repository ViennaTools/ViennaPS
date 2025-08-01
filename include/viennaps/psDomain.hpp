#pragma once

#include "psDomainSetup.hpp"
#include "psMaterials.hpp"
#include "psPreCompileMacros.hpp"
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

inline constexpr std::string_view version = "3.6.1";
inline constexpr int versionMajor = static_cast<int>(version[0] - '0');
inline constexpr int versionMinor = static_cast<int>(version[2] - '0');
inline constexpr int versionPatch = static_cast<int>(version[4] - '0');

using namespace viennacore;

enum class MetaDataLevel {
  NONE = 0,    // No metadata add to ouptut
  GRID = 1,    // Domain-specific metadata (grid delta, boundary conditions)
  PROCESS = 2, // Process-specific metadata (e.g., process parameters)
  FULL = 3     // Full metadata including all available information (advection
               // parameters, ray tracing parameters, etc.)
};

// This class represents all materials in the simulation domain.
// It contains Level-Sets for an accurate surface representation
// and a cell-based structure for the storage of volume information.
// These structures are used depending on the process applied to the material.
// Processes may use one of either structure or both.
//
// Level-Sets in the domain automatically wrap all lower domains when inserted.
// If specified, each Level-Set is assigned a specific material,
// which can be used in a process to implement material specific rates or
// similar.
template <class NumericType, int D> class Domain {
public:
  using lsDomainType = SmartPointer<viennals::Domain<NumericType, D>>;
  using lsDomainsType = std::vector<lsDomainType>;
  using csDomainType = SmartPointer<viennacs::DenseCellSet<NumericType, D>>;
  using MaterialMapType = SmartPointer<MaterialMap>;
  using MetaDataType =
      std::unordered_map<std::string, std::vector<NumericType>>;
  using Setup = DomainSetup<NumericType, D>;

  static constexpr char materialIdsLabel[] = "MaterialIds";
  static MetaDataLevel useMetaData;

private:
  Setup setup_;
  lsDomainsType levelSets_;
  csDomainType cellSet_ = nullptr;
  MaterialMapType materialMap_ = nullptr;
  MetaDataType metaData_;

public:
  // Default constructor.
  Domain() = default;

  // Deep copy constructor.
  explicit Domain(SmartPointer<Domain> domain) { deepCopy(domain); }

  // Constructor for domain with a single initial Level-Set.
  explicit Domain(lsDomainType levelSet) {
    setup_.init(levelSet->getGrid());
    levelSets_.push_back(levelSet);
    initMetaData();
  }

  // Constructor for domain with multiple initial Level-Sets.
  explicit Domain(lsDomainsType levelSets) : levelSets_(levelSets) {
    setup_.init(levelSets.back()->getGrid());
    initMetaData();
  }

  // Sets up domain in with primary direction y in 2D and z in 3D
  Domain(NumericType gridDelta, NumericType xExtent,
         BoundaryType boundary = BoundaryType::REFLECTIVE_BOUNDARY)
      : setup_(gridDelta, xExtent, 0.0, boundary) {
    initMetaData();
  }

  // Sets up domain in with primary direction y in 2D and z in 3D
  // In 2D yExtent is ignored.
  Domain(NumericType gridDelta, NumericType xExtent, NumericType yExtent = 0.0,
         BoundaryType boundary = BoundaryType::REFLECTIVE_BOUNDARY)
      : setup_(gridDelta, xExtent, yExtent, boundary) {
    initMetaData();
  }

  Domain(double bounds[2 * D], BoundaryType boundaryConditions[D],
         NumericType gridDelta = 1.0)
      : setup_(bounds, boundaryConditions, gridDelta) {
    initMetaData();
  }

  // Convenience function to create a new domain.
  template <class... Args> static auto New(Args &&...args) {
    return SmartPointer<Domain>::New(std::forward<Args>(args)...);
  }

  explicit Domain(const Setup &setup) : setup_(setup) { initMetaData(); }

  void setup(const Setup &setup) {
    setup_ = setup;
    initMetaData();
  }

  void setup(NumericType gridDelta, NumericType xExtent,
             NumericType yExtent = 0,
             BoundaryType boundary = BoundaryType::REFLECTIVE_BOUNDARY) {
    setup_ = Setup(gridDelta, xExtent, yExtent, boundary);
    initMetaData();
  }

  static void
  enableMetaData(const MetaDataLevel level = MetaDataLevel::PROCESS) {
    useMetaData = level;
  }

  static void disableMetaData() { useMetaData = MetaDataLevel::NONE; }

  // Create a deep copy of all Level-Sets and the Cell-Set from the passed
  // domain.
  void deepCopy(SmartPointer<Domain> domain) {

    clear();
    setup_ = domain->setup_;
    metaData_ = domain->metaData_;

    // Copy all Level-Sets.
    for (auto &ls : domain->levelSets_) {
      levelSets_.push_back(lsDomainType::New(ls));
    }

    // Copy material map.
    if (domain->materialMap_) {
      materialMap_ = MaterialMapType::New();
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
    if (levelSets_.empty() && setup_.gridDelta() == 0.0) {
      setup_.init(levelSet->getGrid());
      initMetaData();
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
      initMetaData();
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
      materialMap_ = MaterialMapType::New();
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
          .addError("Cannot duplicate Level-Set in empty domain.")
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
      auto newMatMap = MaterialMapType::New();
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
          .addError("Cannot remove Level-Set at index " + std::to_string(idx) +
                    ". Index out of bounds.")
          .print();
      return;
    }

    if (materialMap_) {
      auto newMatMap = MaterialMapType::New();
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

  void setMaterialMap(const MaterialMapType &passedMaterialMap) {
    materialMap_ = passedMaterialMap;
    materialMapCheck();
  }

  // Set the material of a specific Level-Set in the domain.
  void setMaterial(unsigned int lsId, const Material material) {
    if (materialMap_) {
      materialMap_ = MaterialMapType::New();
    }
    materialMap_->setMaterialAtIdx(lsId, material);
    materialMapCheck();
  }

  // Add metadata to the domain.
  // The metadata is stored as a key-value pair, where the key is a string and
  // the value is a vector of NumericType values.
  // This can be used to store additional information about the domain, such as
  // simulation parameters, process information, or any other relevant data.
  void addMetaData(const std::string &key,
                   const std::vector<NumericType> &values) {
    metaData_[key] = values;
  }

  void addMetaData(const std::string &key, NumericType value) {
    metaData_[key] = std::vector<NumericType>{value};
  }

  void addMetaData(const MetaDataType &metaData) {
    for (const auto &pair : metaData) {
      metaData_[pair.first] = pair.second;
    }
  }

  // Returns the top Level-Set (surface) in the domain.
  auto &getSurface() const { return levelSets_.back(); }

  // Returns a vector with all Level-Sets in the domain.
  auto &getLevelSets() const { return levelSets_; }

  auto getNumberOfLevelSets() const {
    return static_cast<unsigned int>(levelSets_.size());
  }

  // Returns the material map which contains the specified material for each
  // Level-Set in the domain.
  auto &getMaterialMap() const { return materialMap_; }

  auto &getCellSet() const { return cellSet_; }

  // Returns the underlying HRLE grid of the top Level-Set in the domain.
  auto &getGrid() const { return setup_.grid(); }

  auto getGridDelta() const { return setup_.gridDelta(); }

  auto &getSetup() { return setup_; }

  auto getMetaData() const { return metaData_; }

  // Returns the bounding box of the top Level-Set in the domain.
  // [min, max][x, y, z]
  auto getBoundingBox() const {
    std::array<Vec3D<NumericType>, 2> boundingBox;
    if (levelSets_.empty()) {
      Logger::getInstance()
          .addWarning("No Level-Sets in domain. Returning empty bounding box.")
          .print();
      return boundingBox;
    }
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

  void print(std::ostream &out = std::cout, bool hrle = false) const {
    constexpr std::string_view separator =
        "*****************************************\n";
    out << "Process Simulation Domain:\n" << separator;
    out << "Number of Level-Sets: " << levelSets_.size() << "\n";
    if (materialMap_) {
      out << "Materials:\n";
      for (std::size_t i = 0; i < materialMap_->size(); i++) {
        out << "\t" << i << ": "
            << MaterialMap::getMaterialName(materialMap_->getMaterialAtIdx(i))
            << "\n";
      }
    } else {
      out << "No Material Map available.\n";
    }
    auto bb = getBoundingBox();
    out << "Bounding Box: [" << bb[0][0] << ", " << bb[0][1] << ", " << bb[0][2]
        << "] - [" << bb[1][0] << ", " << bb[1][1] << ", " << bb[1][2] << "]\n"
        << separator;
    if (hrle) {
      for (auto &ls : levelSets_) {
        ls->print();
      }
      out << separator;
    }
    if (!metaData_.empty()) {
      out << "Meta Data:\n";
      for (const auto &pair : metaData_) {
        out << "\t" << pair.first << ": ";
        for (const auto &value : pair.second) {
          out << value << " ";
        }
        out << "\n";
      }
      out << separator;
    }
  }

  // Save the level set as a VTK file.
  void saveLevelSetMesh(const std::string &fileName, int width = 1) {
    for (int i = 0; i < levelSets_.size(); i++) {
      auto mesh = viennals::Mesh<NumericType>::New();
      viennals::Expand<NumericType, D>(levelSets_.at(i), width).apply();
      viennals::ToMesh<NumericType, D>(levelSets_.at(i), mesh).apply();
      viennals::VTKWriter<NumericType> writer(
          mesh, fileName + "_layer" + std::to_string(i) + ".vtp");
      writer.setMetaData(metaData_);
      writer.apply();
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
    viennals::VTKWriter<NumericType> writer(mesh, fileName);
    writer.setMetaData(metaData_);
    writer.apply();
  }

  // Save the domain as a volume mesh
  void saveVolumeMesh(std::string fileName,
                      double wrappingLayerEpsilon = 1e-2) const {
    viennals::WriteVisualizationMesh<NumericType, D> writer;
    writer.setFileName(fileName);
    writer.setWrappingLayerEpsilon(wrappingLayerEpsilon);
    for (auto &ls : levelSets_) {
      writer.insertNextLevelSet(ls);
    }
    if (materialMap_)
      writer.setMaterialMap(materialMap_->getMaterialMap());
    writer.setMetaData(metaData_);
    writer.apply();
  }

  void saveHullMesh(std::string fileName,
                    double wrappingLayerEpsilon = 1e-2) const {
    viennals::WriteVisualizationMesh<NumericType, D> writer;
    writer.setFileName(fileName);
    writer.setWrappingLayerEpsilon(wrappingLayerEpsilon);
    writer.setExtractHullMesh(true);
    writer.setExtractVolumeMesh(false);
    for (auto &ls : levelSets_) {
      writer.insertNextLevelSet(ls);
    }
    if (materialMap_)
      writer.setMaterialMap(materialMap_->getMaterialMap());
    writer.setMetaData(metaData_);
    writer.apply();
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
      materialMap_ = MaterialMapType::New();
    clearMetaData(true);
  }

  void clearMetaData(bool clearDomainData = false) {
    metaData_.clear();
    if (!clearDomainData)
      initMetaData();
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

  void initMetaData() {
    if (static_cast<int>(useMetaData) > 0) {
      metaData_["Grid Delta"] = std::vector<NumericType>{setup_.gridDelta()};
      std::vector<NumericType> boundaryConds(D);
      for (int i = 0; i < D; i++) {
        boundaryConds[i] = static_cast<NumericType>(setup_.boundaryCons()[i]);
      }
      metaData_["Boundary Conditions"] = boundaryConds;
    }
  }
};

template <class NumericType, int D>
MetaDataLevel Domain<NumericType, D>::useMetaData = MetaDataLevel::NONE;

PS_PRECOMPILE_PRECISION_DIMENSION(Domain)

} // namespace viennaps
