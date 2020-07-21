#ifndef CS_FROM_LEVEL_SET_HPP
#define CS_FROM_LEVEL_SET_HPP

// #include <set>
// #include <unordered_set>

#include <hrleFillDomainFromPointList.hpp>
#include <hrleFillDomainWithSignedDistance.hpp>
#include <hrleVectorType.hpp>
#include <hrleSparseMultiIterator.hpp>

#include <lsCalculateNormalVectors.hpp>
// #include <lsConvexHull.hpp>

#include <csDomain.hpp>
#include <csSimpleConversion.hpp>

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
  using DataDomainType = typename LSType::value_type::element_type::DomainType;
  using NumericType = typename LSType::value_type::element_type::ValueType;

  LSType levelSets;
  CSType cellSet = nullptr;
  csFromLevelSetsEnum conversionType =
      csFromLevelSetsEnum::SIMPLE; // TODO change to lookup
  bool calculateFillingFraction = true;
  static constexpr int D = LSType::value_type::element_type::dimensions;

  template <class ConversionType> void convert() {

    auto &grid = levelSets[0]->getGrid();
    auto newCSDomain = CSType::New(grid, cellSet->getBackGroundValue(),
                                   cellSet->getEmptyValue());
    auto &newDomain = newCSDomain->getDomain();
    auto &domain = levelSets.back()->getDomain();

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

      // // an iterator for each level set
      // std::vector<hrleConstSparseStarIterator<DataDomainType>> iterators;
      // for (auto it = levelSets.begin(); it != levelSets.end(); ++it) {
      //   iterators.push_back(
      //       hrleConstSparseStarIterator<DataDomainType>((*it)->getDomain()));
      // }

      // std::cout << "Inside convert: " << std::endl;
      // domain.print();

      // for (iterators.back().goToIndices(startVector);
      //      iterators.back().getIndices() < endVector; iterators.back().next()) {
      for(hrleConstSparseIterator<DataDomainType> it(domain, startVector); it.getStartIndices() < endVector; it.next()) {
      // for(hrleConstSparseMultiIterator<DataDomainType> it(domain); it.getIndices() < endVector; it.next()) {
        // std::cout << it.getStartIndices();
        // check top levelset first, if there is no surface inside, skip point

        // skip this voxel if there is no surface inside
        if (!it.isDefined() ||
            std::abs(it.getValue()) > 0.5) {
          auto undefinedValue = (it.getValue() > 0)
                                    ? cellSet->getEmptyValue()
                                    : cellSet->getBackGroundValue();
          // insert an undefined point to create correct hrle structure
          newDomain.insertNextUndefinedPoint(p, it.getStartIndices(),
                                             undefinedValue);
        } else {
          CellType cell;

          // convert LS value to filling Fraction
          float fillingFraction = 0.5 - it.getValue();
              //ConversionType(iterators.back()).getFillingFraction();
          cell.setInitialFillingFraction(fillingFraction);
          newDomain.insertNextDefinedPoint(p, it.getStartIndices(), cell);
        }
      } // end of ls loop
    }   // end of parallel

    // distribute evenly across segments and copy
    newDomain.finalize();
    newDomain.segment();
    // copy new domain into old csdomain
    cellSet->deepCopy(newCSDomain);

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
    if (levelSets.empty()) {
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
      convert<csSimpleConversion<NumericType, D>>();
      break;
    }
  } // apply()
};

#endif // CS_FROM_LEVEL_SET_HPP
