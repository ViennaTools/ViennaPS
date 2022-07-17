#ifndef PS_DOMAIN_HPP
#define PS_DOMAIN_HPP

#include <lsDomain.hpp>
#include <lsMakeGeometry.hpp>
#include <lsToDiskMesh.hpp>
#include <lsToSurfaceMesh.hpp>
#include <lsVTKWriter.hpp>

#include <csDomain.hpp>
#include <csFromLevelSets.hpp>
#include <csToLevelSets.hpp>

#include <psSmartPointer.hpp>

/**
  This class represents all materials in the simulation domain.
  It contains level sets for the accurate surface representation
  and a cell-based structure for the storage of volume information.
  These structures are used depending on the process applied to the material.
  Processes may use one of either structures or both.
*/
template <class CellType, class NumericType = float, int D = 3> class psDomain {
public:
  typedef psSmartPointer<lsDomain<NumericType, D>> lsDomainType;
  typedef psSmartPointer<std::vector<lsDomainType>> lsDomainsType;
  typedef psSmartPointer<csDomain<CellType, D>> csDomainType;

private:
  lsDomainsType levelSets = nullptr;
  csDomainType cellSet = nullptr;
  bool syncData = true;

public:
  psDomain(double gridDelta = 1.0, bool sync = true) : syncData(sync) {
    CellType backGroundCell;
    backGroundCell.setInitialFillingFraction(1.0);
    CellType emptyCell;
    emptyCell.setInitialFillingFraction(0.0);
    *this = psDomain(gridDelta, backGroundCell, emptyCell);
  }

  psDomain(double gridDelta, CellType backGroundCell, bool sync = true)
      : syncData(sync) {
    CellType emptyCell;
    emptyCell.setInitialFillingFraction(0.0);
    *this = psDomain(gridDelta, backGroundCell, emptyCell);
  }

  /// If no other geometry is passed to psDomain,
  /// a level set describing a plane substrate will be instantiatied.
  psDomain(double gridDelta, CellType backGroundCell, CellType emptyCell,
           bool sync = true)
      : syncData(sync) {
    double bounds[2 * D] = {-20, 20, -20, 20};
    if constexpr (D == 3) {
      bounds[4] = -20;
      bounds[5] = 20;
    }

    typename lsDomain<NumericType, D>::BoundaryType boundaryCons[D];
    for (unsigned i = 0; i < D - 1; ++i) {
      boundaryCons[i] =
          lsDomain<NumericType, D>::BoundaryType::REFLECTIVE_BOUNDARY;
    }
    boundaryCons[D - 1] =
        lsDomain<NumericType, D>::BoundaryType::INFINITE_BOUNDARY;

    auto substrate = psSmartPointer<lsDomain<NumericType, D>>::New(
        bounds, boundaryCons, gridDelta);
    NumericType origin[3] = {0., 0., 0.};
    NumericType planeNormal[3] = {0, D == 2, D == 3};

    // set up the level set
    lsMakeGeometry<NumericType, D>(
        substrate,
        psSmartPointer<lsPlane<NumericType, D>>::New(origin, planeNormal))
        .apply();
    // push level set into list
    levelSets = lsDomainsType::New();
    levelSets->push_back(substrate);

    cellSet =
        csDomainType::New(substrate->getGrid(), backGroundCell, emptyCell);

    // generate the cell set from the levelset
    if (syncData) {
      generateCellSet();
    }
  }

  psDomain(lsDomainType passedLevelSet) {
    levelSets = lsDomainsType::New();
    levelSets->push_back(passedLevelSet);
    cellSet = csDomainType::New(passedLevelSet->getGrid());
    // generate CellSet
    if (syncData) {
      generateCellSet();
    }
  }

  psDomain(csDomainType passedCellSet) {
    cellSet = csDomainType::New(passedCellSet);
    levelSets = lsDomainsType::New(passedCellSet->getGrid());
    if (syncData) {
      generateLevelSets();
    }
  }

  void deepCopy(psSmartPointer<psDomain> passedDomain) {
    levelSets->resize(passedDomain->levelSets->size());
    for (unsigned i = 0; i < levelSets->size(); ++i) {
      levelSets[i]->deepCopy(passedDomain->levelSets[i]);
    }
    cellSet->deepCopy(passedDomain->cellSet);
    syncData = passedDomain->syncData;
  }

  void insertNextLevelSet(lsDomainType passedLevelSet,
                          bool wrapLowerLevelSet = true) {
    if (!levelSets->empty() && wrapLowerLevelSet) {
      lsBooleanOperation<NumericType, D>(levelSets->front(), passedLevelSet,
                                         lsBooleanOperationEnum::UNION)
          .apply();
    }
    levelSets->push_back(passedLevelSet);

    if (syncData) {
      generateCellSet();
    }
  }

  void generateCellSet(bool calculateFillingFraction = true) {
    csFromLevelSets<lsDomainsType, csDomainType>(levelSets, cellSet,
                                                 calculateFillingFraction)
        .apply();
  }

  void generateLevelSets() {
    csToLevelSets<lsDomainsType, csDomainType>(levelSets, cellSet).apply();
  }

  auto &getLevelSets() { return levelSets; }

  auto &getCellSet() { return cellSet; }

  auto &getGrid() { return levelSets->back()->getGrid(); }

  void setSyncData(bool sync) { syncData = sync; }

  bool getSyncData() { return syncData; }

  void print() {
    std::cout << "Process Simulation Domain:" << std::endl;
    std::cout << "**************************" << std::endl;
    for (auto &ls : *levelSets) {
      ls->print();
    }
    cellSet->print();
    std::cout << "**************************" << std::endl;
  }
};

#endif // PS_DOMAIN_HPP
