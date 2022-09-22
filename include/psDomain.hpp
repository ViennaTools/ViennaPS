#ifndef PS_DOMAIN_HPP
#define PS_DOMAIN_HPP

#include <lsBooleanOperation.hpp>
#include <lsDomain.hpp>
#include <lsMakeGeometry.hpp>
#include <lsToDiskMesh.hpp>
#include <lsToSurfaceMesh.hpp>
#include <lsVTKWriter.hpp>

#include <csDenseCellSet.hpp>

#include <psSmartPointer.hpp>

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

private:
  lsDomainsType levelSets = nullptr;
  csDomainType cellSet = nullptr;
  bool useCellSet = false;
  NumericType cellSetDepth = 0.;

public:
  psDomain(bool passedUseCellSet = false) : useCellSet(passedUseCellSet) {
    levelSets = lsDomainsType::New();
    if (useCellSet) {
      cellSet = csDomainType::New();
    }
  }

  psDomain(lsDomainType passedLevelSet, bool passedUseCellSet = false)
      : useCellSet(passedUseCellSet) {
    levelSets = lsDomainsType::New();
    levelSets->push_back(passedLevelSet);
    // generate CellSet
    if (useCellSet) {
      cellSet = csDomainType::New(levelSets);
    }
  }

  psDomain(lsDomainsType passedLevelSets, bool passedUseCellSet = false,
           const NumericType passedDepth = 0.)
      : useCellSet(passedUseCellSet) {
    levelSets = passedLevelSets;
    // generate CellSet
    if (useCellSet) {
      cellSetDepth = passedDepth;
      cellSet = csDomainType::New(levelSets, cellSetDepth);
    }
  }

  void deepCopy(psSmartPointer<psDomain> passedDomain) {
    levelSets->resize(passedDomain->levelSets->size());
    for (unsigned i = 0; i < levelSets->size(); ++i) {
      levelSets[i]->deepCopy(passedDomain->levelSets[i]);
    }
    useCellSet = passedDomain->useCellSet;
    if (useCellSet) {
      cellSetDepth = passedDomain->cellSetDepth;
      cellSet->fromLevelSets(passedDomain->levelSets, cellSetDepth);
    }
  }

  void insertNextLevelSet(lsDomainType passedLevelSet,
                          bool wrapLowerLevelSet = true) {
    if (!levelSets->empty() && wrapLowerLevelSet) {
      lsBooleanOperation<NumericType, D>(passedLevelSet, levelSets->front(),
                                         lsBooleanOperationEnum::UNION)
          .apply();
    }
    levelSets->push_back(passedLevelSet);
  }

  void generateCellSet(const NumericType depth = 0.) {
    useCellSet = true;
    cellSetDepth = depth;
    if (cellSet == nullptr) {
      cellSet = csDomainType::New();
    }
    cellSet->fromLevelSets(levelSets, cellSetDepth);
  }

  auto &getLevelSets() { return levelSets; }

  auto &getCellSet() { return cellSet; }

  auto &getGrid() { return levelSets->back()->getGrid(); }

  void setUseCellSet(bool useCS) { useCellSet = useCS; }

  bool getUseCellSet() { return useCellSet; }

  void print() {
    std::cout << "Process Simulation Domain:" << std::endl;
    std::cout << "**************************" << std::endl;
    for (auto &ls : *levelSets) {
      ls->print();
    }
    std::cout << "**************************" << std::endl;
  }

  void printSurface(std::string name) {
    auto mesh = psSmartPointer<lsMesh<NumericType>>::New();
    lsToSurfaceMesh<NumericType, D>(levelSets->back(), mesh).apply();

    lsVTKWriter<NumericType>(mesh, name).apply();
  }
};

#endif // PS_DOMAIN_HPP
