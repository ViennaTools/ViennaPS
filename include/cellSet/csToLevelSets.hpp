#ifndef CS_TO_LEVEL_SET_HPP
#define CS_TO_LEVEL_SET_HPP

#include <hrleFillDomainFromPointList.hpp>
#include <hrleFillDomainWithSignedDistance.hpp>
#include <hrleVectorType.hpp>

#include <lsBooleanOperation.hpp>
#include <lsCalculateNormalVectors.hpp>
#include <lsExpand.hpp>
#include <lsMessage.hpp>

#include <psSmartPointer.hpp>

/// Enumeration for the different types of conversion
enum struct csToLevelSetsEnum : unsigned {
  ANALYTICAL = 0,
  LOOKUP = 1,
  SIMPLE = 2,
};

/// This algorithm converts a cellSet into a levelSet by
/// calculating the surface normal from neighbouring cells
/// and using the filling fraction to find a point on the surface,
/// which can then be converted to a level set value.
template <class LSType, class CSType> class csToLevelSets {
  using CellType = typename CSType::element_type::ValueType;
  using LevelSetType = typename LSType::element_type::value_type::element_type;
  using DataDomainType = typename LevelSetType::DomainType;
  using NumericType = typename LevelSetType::ValueType;

  LSType levelSets = nullptr;
  CSType cellSet = nullptr;
  csToLevelSetsEnum conversionType =
      csToLevelSetsEnum::SIMPLE; // TODO change to lookup
  static constexpr int D = LevelSetType::dimensions;
  static constexpr typename LevelSetType::ValueType eps = 1e-6;

  void convertSimple() {
    // typedef typename CSType::ValueType CellType;
    auto &grid = cellSet->getGrid();
    auto &domain = cellSet->getDomain();

    // auto newLSDomain = LSType::New(grid);
    // typename LSType::DomainType &newDomain = newLSDomain.getDomain();
    // newDomain.initialize(domain.getNewSegmentation(),
    // domain.getAllocation());

    // Initialize the correct number of level sets
    const unsigned numberOfMaterials = cellSet->getNumberOfMaterials();
    levelSets->resize(numberOfMaterials);
    for (auto &it : *levelSets) {
      it = LSType::element_type::value_type::New(grid);
      it->getDomain().initialize(domain.getNewSegmentation(),
                                 domain.getAllocation());
    }

// go over each point and calculate the filling fraction
#pragma omp parallel num_threads(domain.getNumberOfSegments())
    {
      int p = 0;
#ifdef _OPENMP
      p = omp_get_thread_num();
#endif

      hrleVectorType<hrleIndexType, D> startVector =
          (p == 0) ? grid.getMinGridPoint() : domain.getSegmentation()[p - 1];

      hrleVectorType<hrleIndexType, D> endVector =
          (p != static_cast<int>(domain.getNumberOfSegments() - 1))
              ? domain.getSegmentation()[p]
              : grid.incrementIndices(grid.getMaxGridPoint());

      for (hrleConstSparseIterator<typename CSType::element_type::DomainType>
               it(domain, startVector);
           it.getStartIndices() < endVector; it.next()) {

        // map containing all material IDs and filling fractions
        auto &materialMap = it.getValue().getMaterialFractions();

        // go over all possible material ids
        const bool isDefined = it.isDefined();
        typename CellType::MaterialFractionType::mapped_type cumulativeFF = 0.0;
        for (unsigned i = 0; i < numberOfMaterials; ++i) {
          auto &dataDomain = levelSets->at(i)->getDomain();
          auto material = materialMap.find(i);

          // ff is defined for material
          if (material != materialMap.end()) {
            cumulativeFF += material->second;
            if (isDefined && (cumulativeFF > eps) &&
                (cumulativeFF < (1.0 - eps))) { // defined point in cellSet
              dataDomain.insertNextDefinedPoint(p, it.getStartIndices(),
                                                0.5 - cumulativeFF);
            } else { // currently on undefined point in cellSet
              if (cumulativeFF < eps) { // ff is empty
                dataDomain.insertNextUndefinedPoint(p, it.getStartIndices(),
                                                    LevelSetType::POS_VALUE);
              } else if (cumulativeFF > (1.0 - eps)) { // ff is full
                dataDomain.insertNextUndefinedPoint(p, it.getStartIndices(),
                                                    LevelSetType::NEG_VALUE);
                cumulativeFF = 1.0;
              } else { // invalid ff for undefined point
                lsMessage::getInstance()
                    .addWarning("Background cell value should not have a "
                                "filling fraction other than 0 or 1!")
                    .print();
              }
            }
          } else if (cumulativeFF > eps) {    // if material is not defined, but
                                              // material below was
            if (cumulativeFF > (1.0 - eps)) { // ff is full
              dataDomain.insertNextUndefinedPoint(p, it.getStartIndices(),
                                                  LevelSetType::NEG_VALUE);
            } else { // take the same value as material below
              dataDomain.insertNextDefinedPoint(p, it.getStartIndices(),
                                                0.5 - cumulativeFF);
            }
          } else { // id is not found --> ff must be 0
            dataDomain.insertNextUndefinedPoint(p, it.getStartIndices(),
                                                LevelSetType::POS_VALUE);
          }
        }
      }
    }

    // finalize all level sets
    for (unsigned i = 0; i < levelSets->size(); ++i) {
      auto &currentLS = levelSets->at(i);
      auto &dataDomain = currentLS->getDomain();
      // distribute evenly across segments and copy
      dataDomain.finalize();
      dataDomain.segment();
      currentLS->setLevelSetWidth(1);

      // pad to width of 2, so it is normalised again
      lsExpand<typename LevelSetType::ValueType, D>(currentLS, 2).apply();

      //       #ifndef NDEBUG
      //       auto mesh = lsSmartPointer<lsMesh>::New();
      //       lsToMesh<typename LevelSetType::ValueType, D>(currentLS,
      //       mesh).apply(); lsVTKWriter(mesh, lsFileFormatEnum::VTP,
      //       "csToLevelSets_DEBUG-" + std::to_string(i) + ".vtp").apply();
      //       lsToSurfaceMesh<typename LevelSetType::ValueType, D>(currentLS,
      //       mesh).apply(); lsVTKWriter(mesh, lsFileFormatEnum::VTP,
      //       "csToLevelSets_DEBUG-surface-" + std::to_string(i) +
      //       ".vtp").apply();
      // #endif
    }
  }

public:
  csToLevelSets() {}

  csToLevelSets(LSType passedLevelSets) : levelSets(passedLevelSets) {}

  csToLevelSets(LSType passedLevelSets, CSType passedCellSet)
      : levelSets(passedLevelSets), cellSet(passedCellSet) {}

  void setLevelSets(LSType passedLevelSets) { levelSets = passedLevelSets; }

  void setCellSet(CSType passedCellSet) { cellSet = passedCellSet; }

  void apply() {
    if (levelSets == nullptr) {
      lsMessage::getInstance()
          .addWarning("No level set container was passed to csToLevelSets.")
          .print();
      return;
    }
    if (cellSet == nullptr) {
      lsMessage::getInstance()
          .addWarning("No cell set was passed to csToLevelSets.")
          .print();
      return;
    }

    // remove all old level sets
    levelSets->clear();

    switch (conversionType) {
    case csToLevelSetsEnum::ANALYTICAL:
      // convertAnalytical
      break;
    case csToLevelSetsEnum::LOOKUP:
      // convertLookup
      break;
    case csToLevelSetsEnum::SIMPLE:
      convertSimple();
      break;
    }
  }
};

#endif // CS_TO_LEVEL_SET_HPP
