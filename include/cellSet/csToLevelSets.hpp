#ifndef CS_TO_LEVEL_SET_HPP
#define CS_TO_LEVEL_SET_HPP

#include <hrleFillDomainFromPointList.hpp>
#include <hrleFillDomainWithSignedDistance.hpp>
#include <hrleVectorType.hpp>

#include <lsCalculateNormalVectors.hpp>
#include <lsExpand.hpp>

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
  LSType *levelSet = nullptr;
  CSType *cellSet = nullptr;
  csToLevelSetsEnum conversionType =
      csToLevelSetsEnum::SIMPLE; // TODO change to lookup
  static constexpr int D = LSType::dimensions;

  void convertSimple() {
    // typedef typename CSType::ValueType CellType;
    typename CSType::GridType &grid = cellSet->getGrid();
    LSType newLSDomain(grid);
    typename LSType::DomainType &newDomain = newLSDomain.getDomain();
    typename CSType::DomainType &domain = cellSet->getDomain();

    newDomain.initialize(domain.getNewSegmentation(), domain.getAllocation());

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

      for (hrleConstSparseIterator<typename CSType::DomainType> it(domain,
                                                                   startVector);
           it.getStartIndices() < endVector; it.next()) {

        // skip this voxel if there is no plane inside
        if (!it.isDefined()) {
          auto undefinedValue = (it.getValue().getFillingFraction() ==
                                 cellSet->getEmptyValue().getFillingFraction())
                                    ? LSType::POS_VALUE
                                    : LSType::NEG_VALUE;
          // insert an undefined point to create correct hrle structure
          newDomain.insertNextUndefinedPoint(p, it.getStartIndices(),
                                             undefinedValue);
          continue;
        }

        // generate the normal vector by considering neighbours
        auto fillingFraction = it.getValue().getFillingFraction();

        newDomain.insertNextDefinedPoint(p, it.getStartIndices(),
                                         0.5 - fillingFraction);
      }
    }

    // distribute evenly across segments and copy
    newDomain.finalize();
    newDomain.segment();
    // copy new domain into old lsdomain
    levelSet->getDomain().deepCopy(grid, newDomain);
    levelSet->setLevelSetWidth(1);

    // pad to width of 2, so it is normalised again
    lsExpand<typename LSType::ValueType, D>(*levelSet, 2).apply();
  }

public:
  csToLevelSets() {}

  csToLevelSets(LSType &passedLevelSet) : levelSet(&passedLevelSet) {}

  csToLevelSets(LSType &passedLevelSet, CSType &passedCellSet)
      : levelSet(&passedLevelSet), cellSet(&passedCellSet) {}

  void setLevelSet(LSType &passedLevelSet) { levelSet = &passedLevelSet; }

  void setCellSet(CSType &passedCellSet) { cellSet = &passedCellSet; }

  void apply() {
    if (levelSet == nullptr) {
      lsMessage::getInstance()
          .addWarning("No level set was passed to csFromLevelSet.")
          .print();
      return;
    }
    if (cellSet == nullptr) {
      lsMessage::getInstance()
          .addWarning("No cell set was passed to csFromLevelSet.")
          .print();
      return;
    }

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
