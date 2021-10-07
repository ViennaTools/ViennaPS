#ifndef CS_FROM_LEVEL_SET_HPP
#define CS_FROM_LEVEL_SET_HPP

#include <unordered_set>

#include <hrleFillDomainFromPointList.hpp>
#include <hrleFillDomainWithSignedDistance.hpp>
#include <hrleSparseMultiIterator.hpp>
#include <hrleVectorType.hpp>

#include <lsCalculateNormalVectors.hpp>
// #include <lsConvexHull.hpp>

#include <csDomain.hpp>

/// Enumeration for the different types of conversion
enum struct csFromLevelSetsEnum : unsigned {
  ANALYTICAL = 0,
  LOOKUP = 1,
  SIMPLE = 2,
};

/// This class constructs a csDomain from an lsDomain.
/// Filling fraction can be calculated by cutting each voxel
/// by the plane of an LS disk (normal and ls value) and computing the volume.
template <class LSType, class CSType> class csFromLevelSets {
  using CellType = typename CSType::element_type::ValueType;
  using DataDomainType =
      typename LSType::element_type::value_type::element_type::DomainType;
  using NumericType =
      typename LSType::element_type::value_type::element_type::ValueType;

  LSType levelSets;
  CSType cellSet = nullptr;
  csFromLevelSetsEnum conversionType =
      csFromLevelSetsEnum::SIMPLE; // TODO change to lookup
  bool calculateFillingFraction = true;
  static constexpr int D =
      LSType::element_type::value_type::element_type::dimensions;

  void convertSimple() {

    auto &grid = levelSets->at(0)->getGrid();
    auto newCSDomain = CSType::New(grid, cellSet->getBackGroundValue(),
                                   cellSet->getEmptyValue());
    auto &newDomain = newCSDomain->getDomain();
    auto &domain = levelSets->back()->getDomain();

    newDomain.initialize(domain.getNewSegmentation(), domain.getAllocation());

    // record different number of materials
    std::vector<std::unordered_set<unsigned>> materialSets;
    materialSets.resize(domain.getNumberOfSegments());

// go over each point and calculate the filling fraction
#pragma omp parallel num_threads(domain.getNumberOfSegments())
    {
      int p = 0;
#ifdef _OPENMP
      p = omp_get_thread_num();
#endif

      // collect domain pointers
      std::vector<const DataDomainType *> domains;
      for (auto &it : *levelSets) {
        domains.push_back(&it->getDomain());
      }

      hrleVectorType<hrleIndexType, D> startVector =
          (p == 0) ? grid.getMinGridPoint() : domain.getSegmentation()[p - 1];

      hrleVectorType<hrleIndexType, D> endVector =
          (p != static_cast<int>(domain.getNumberOfSegments() - 1))
              ? domain.getSegmentation()[p]
              : grid.incrementIndices(grid.getMaxGridPoint());

      for (hrleConstSparseMultiIterator<DataDomainType> it(domains);
           it.getIndices() < endVector; it.next()) {

        // skip this voxel if there is no surface inside
        if (!it.isDefined()) {
          CellType cell;
          auto &materialFractions = cell.getMaterialFractions();
          for (unsigned i = 0; i < it.getNumberOfDomains(); ++i) {
            if (it.getIterator(i).getValue() < 0.0) {
              materialFractions.insert(std::make_pair(i, 1.0));
              break;
            }
          }
          if (materialFractions.empty()) {
            materialFractions.insert(std::make_pair(0, 0.0));
          }

          // insert an undefined point to create correct hrle structure
          newDomain.insertNextUndefinedPoint(p, it.getIndices(), cell);

        } else {
          CellType cell;
          auto &materialFractions = cell.getMaterialFractions();
          float lastFillingFraction = 0.0;

          for (unsigned i = 0; i < it.getNumberOfDomains(); ++i) {
            if (lastFillingFraction < 1.0) {
              auto &lsValue = it.getIterator(i).getValue();

              if (std::abs(lsValue) < 0.5) {
                // convert LS value to filling Fraction
                // TODO: if we want some more complex way of converting
                // just pass the lsValue and the normal vector to
                // some other class doing the calculations here
                float fillingFraction = 0.5 - lsValue;
                materialFractions.insert(
                    std::make_pair(i, fillingFraction - lastFillingFraction));
                materialSets[p].insert(i);
                lastFillingFraction = fillingFraction;
              } else if (lsValue <= -0.5) {
                // point is definitely inside material
                materialFractions.insert(
                    std::make_pair(i, 1.0 - lastFillingFraction));
                materialSets[p].insert(i);
                // lastFillingFraction = 1.0;
                break;
              }
            }
          }

          // if there was no actual defined point < 0.5
          if (materialFractions.empty()) {
            auto undefinedValue = cellSet->getEmptyValue();
            // insert an undefined point to create correct hrle structure
            newDomain.insertNextUndefinedPoint(p, it.getIndices(),
                                               undefinedValue);
          } else {
            newDomain.insertNextDefinedPoint(p, it.getIndices(), cell);
          }
        }
      } // end of ls loop
    }   // end of parallel

    for (unsigned i = 1; i < materialSets.size(); ++i) {
      materialSets[0].insert(materialSets[i].begin(), materialSets[i].end());
    }

    // distribute evenly across segments and copy
    newDomain.finalize();
    newDomain.segment();
    // copy new domain into old csdomain
    cellSet->deepCopy(newCSDomain);
    cellSet->setNumberOfMaterials(materialSets[0].size());
  }

public:
  csFromLevelSets() {}

  csFromLevelSets(LSType passedlevelSets) : levelSets(passedlevelSets) {}

  csFromLevelSets(LSType passedlevelSets, CSType passedCellSet, bool cff = true)
      : levelSets(passedlevelSets), cellSet(passedCellSet),
        calculateFillingFraction(cff) {}

  csFromLevelSets(LSType passedlevelSets, CSType passedCellSet,
                  csFromLevelSetsEnum conversionEnum, bool cff = true)
      : levelSets(passedlevelSets), cellSet(passedCellSet),
        conversionType(conversionEnum), calculateFillingFraction(cff) {}

  void setlevelSets(LSType passedlevelSets) { levelSets = passedlevelSets; }

  void setCellSet(CSType passedCellSet) { cellSet = passedCellSet; }

  void setCalculateFillingFraction(bool cff) { calculateFillingFraction = cff; }

  void apply() {
    if (levelSets->empty()) {
      lsMessage::getInstance()
          .addWarning("No level set was passed to csFromLevelSets.")
          .print();
      return;
    }
    if (cellSet == nullptr) {
      lsMessage::getInstance()
          .addWarning("No cell set was passed to csFromLevelSets.")
          .print();
      return;
    }

    switch (conversionType) {
    case csFromLevelSetsEnum::ANALYTICAL:
      // convertAnalytical
      break;
    case csFromLevelSetsEnum::LOOKUP:
      // convertLookup
      break;
    case csFromLevelSetsEnum::SIMPLE:
      convertSimple();
      break;
    }
  } // apply()
};

#endif // CS_FROM_LEVEL_SET_HPP
