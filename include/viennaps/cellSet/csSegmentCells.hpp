#pragma once

#include "csDenseCellSet.hpp"

template <class NumericType, int D> class csSegmentCells {
  lsSmartPointer<csDenseCellSet<NumericType, D>> cellSet = nullptr;
  std::string cellTypeString = "CellType";
  int bulkMaterial = -1;

public:
  csSegmentCells(
      const lsSmartPointer<csDenseCellSet<NumericType, D>> &passedCellSet)
      : cellSet(passedCellSet) {}

  template <class Material>
  csSegmentCells(
      const lsSmartPointer<csDenseCellSet<NumericType, D>> &passedCellSet,
      const std::string cellTypeString, const Material passedBulkMaterial)
      : cellSet(passedCellSet), cellTypeString(cellTypeString),
        bulkMaterial(static_cast<int>(passedBulkMaterial)) {}

  void setCellSet(
      const lsSmartPointer<csDenseCellSet<NumericType, D>> &passedCellSet) {
    cellSet = passedCellSet;
  }

  void setCellTypeString(const std::string passedCellTypeString) {
    cellTypeString = passedCellTypeString;
  }

  template <class Material>
  void setBulkMaterial(const Material passedBulkMaterial) {
    bulkMaterial = static_cast<int>(passedBulkMaterial);
  }

  void apply() {
    auto cellType = cellSet->addScalarData(cellTypeString, -1.);
    cellSet->buildNeighborhood();
    auto materials = cellSet->getScalarData("Material");

#pragma omp parallel for
    for (int i = 0; i < materials->size(); ++i) {
      if (static_cast<int>(materials->at(i)) != bulkMaterial) {
        auto neighbors = cellSet->getNeighbors(i);
        for (auto n : neighbors) {
          if (n >= 0 &&
              static_cast<int>(materials->at(n)) == bulkMaterial) {
            cellType->at(i) = 0.;
            break;
          }
        }
      } else {
        cellType->at(i) = 1.;
      }
    }
  }
};
