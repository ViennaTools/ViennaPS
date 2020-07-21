#ifndef PS_DOMAIN_HPP
#define PS_DOMAIN_HPP

#include <lsDomain.hpp>
#include <lsMakeGeometry.hpp>

#include <csDomain.hpp>
#include <csFromLevelSets.hpp>
#include <csToLevelSets.hpp>

/**
  This class represents all materials in the simulation domain.
  It contains level sets for the accurate surface representation
  and a cell-based structure for the storage of volume information.
  These structures are used depending on the process applied to the material.
  Processes may use one of either structures or both.
*/
template <class CellType, class NumericType = float, int D = 3> class psDomain {
public:
  typedef lsSmartPointer<lsDomain<NumericType, D>> lsDomainType;
  typedef std::vector<lsDomainType> lsDomainsType;
  typedef lsSmartPointer<csDomain<CellType, D>> csDomainType;

private:
  lsDomainsType levelSets;
  csDomainType cellSet;

public:
  /// If no other geometry is passed to psDomain,
  /// a level set describing a plane substrate will be instantiatied.
  psDomain(double gridDelta = 1.0, CellType backGroundCell = CellType(),
           CellType emptyCell = CellType()) {
    double bounds[2 * D] = {-20, 20, -20, 20};
    if (D == 3) {
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

    auto substrate = lsSmartPointer<lsDomain<NumericType, D>>::New(
        bounds, boundaryCons, gridDelta);
    NumericType origin[3] = {0., 0., 0.};
    NumericType planeNormal[3] = {0, D == 2, D == 3};

    // set up the level set
    lsMakeGeometry<NumericType, D>(
        substrate,
        lsSmartPointer<lsPlane<NumericType, D>>::New(origin, planeNormal))
        .apply();
    // push level set into list
    levelSets.push_back(substrate);

    cellSet =
        csDomainType::New(substrate->getGrid(), backGroundCell, emptyCell);

    // generate the cell set from the levelset
    // generateCellSet();
  }

  psDomain(lsDomainType passedLevelSet) {
    levelSets.push_back(passedLevelSet);
    cellSet = csDomainType::New(passedLevelSet->getGrid());
  }

  psDomain(csDomain<CellType, D> &passedCellSet) {
    cellSet = csDomainType::New(passedCellSet);
  }

  psDomain(csDomainType passedCellSet) : cellSet(passedCellSet) {}

  void generateCellSet(bool calculateFillingFraction = true) {
    csFromLevelSets<lsDomainsType, csDomainType>(levelSets, cellSet,
                                                 calculateFillingFraction)
        .apply();
  }

  void generateLevelSet() {
    csToLevelSets<lsDomainsType, csDomainType>(levelSets, cellSet).apply();
  }

  auto &getLevelSets() { return levelSets; }

  auto &getCellSet() { return cellSet; }

  auto &getGrid() { return levelSets[0]->getGrid(); }

  void print() {
    std::cout << "Process Simulation Domain:" << std::endl;
    std::cout << "**************************" << std::endl;
    for (auto &ls : levelSets) {
      ls->print();
    }
    cellSet->print();
    std::cout << "**************************" << std::endl;
  }
};

#endif // PS_DOMAIN_HPP
