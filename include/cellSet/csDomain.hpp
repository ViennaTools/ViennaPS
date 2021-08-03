#ifndef CS_DOMAIN_HPP
#define CS_DOMAIN_HPP

#include <hrleDomain.hpp>

#include <cellBase.hpp>
#include <csSmartPointer.hpp>

template <class T, int D> class csDomain {
public:
  typedef T ValueType;
  typedef hrleGrid<D> GridType;
  typedef hrleDomain<T, D> DomainType;
  typedef typename GridType::boundaryType BoundaryType;
  typedef typename std::vector<std::array<double, D>> NormalVectorType;
  NormalVectorType normalVectors;

private:
  GridType grid;
  DomainType domain;
  T backGroundValue;
  T emptyValue;
  unsigned numberOfMaterials = 0;

public:
  static constexpr int dimensions = D;

  csDomain(hrleCoordType gridDelta = 1.0, T backGround = T(), T empty = T())
      : backGroundValue(backGround), emptyValue(empty), numberOfMaterials(1) {
    hrleIndexType gridMin[D], gridMax[D];
    BoundaryType boundaryCons[D];
    for (unsigned i = 0; i < D; ++i) {
      gridMin[i] = 0;
      gridMax[i] = 0;
      boundaryCons[i] = BoundaryType::INFINITE_BOUNDARY;
    }

    grid = GridType(gridMin, gridMax, gridDelta, boundaryCons);
    domain.deepCopy(grid, DomainType(grid, backGroundValue));
  }

  /// construct empty csDomain from the passed grid
  csDomain(const GridType &passedGrid, T backGround = T(), T empty = T())
      : grid(passedGrid), backGroundValue(backGround), emptyValue(empty) {
    domain.deepCopy(grid, DomainType(grid, backGroundValue));
  }

  /// deep copy a csDomain
  csDomain(csSmartPointer<const csDomain<T, D>> passedCellSet)
      : emptyValue(passedCellSet->emptyValue),
        backGroundValue(passedCellSet->backGroundValue) {
    deepCopy(passedCellSet);
  }

  /// deep copy the passed csDomain into this one
  void deepCopy(csSmartPointer<const csDomain<T, D>> passedCellSet) {
    grid = passedCellSet->grid;
    backGroundValue = passedCellSet->backGroundValue;
    emptyValue = passedCellSet->emptyValue;
    domain.deepCopy(grid, passedCellSet->domain);
  }

  /// get reference to the grid on which the cell set is defined
  const GridType &getGrid() const { return grid; }

  /// get mutable reference to the grid on which the cell set is defined
  GridType &getGrid() { return grid; }

  /// get reference to the underlying hrleDomain data structure
  DomainType &getDomain() { return domain; }

  /// get const reference to the underlying hrleDomain data structure
  const DomainType &getDomain() const { return domain; }

  /// returns the number of segments, the levelset is split into.
  /// This is useful for algorithm parallelisation
  unsigned getNumberOfSegments() const { return domain.getNumberOfSegments(); }

  /// returns the number of defined points
  unsigned getNumberOfCells() const { return domain.getNumberOfPoints(); }

  /// set the number of materials currently saved in the cell set
  void setNumberOfMaterials(unsigned number) { numberOfMaterials = number; }

  /// returns the number of different materials currently saved in the cell set
  unsigned getNumberOfMaterials() const { return numberOfMaterials; }

  /// get reference to the normalVectors of all points
  NormalVectorType &getNormalVectors() { return normalVectors; }

  const NormalVectorType &getNormalVectors() const { return normalVectors; }

  /// returns the background value set during initialisation
  T getBackGroundValue() const { return backGroundValue; }

  /// returns the value used for cells, if there is no material
  T getEmptyValue() const { return emptyValue; }

  void print() {
    std::cout << "CellSet" << std::endl;
    domain.print();
  }
};

#endif // CS_DOMAIN_HPP
