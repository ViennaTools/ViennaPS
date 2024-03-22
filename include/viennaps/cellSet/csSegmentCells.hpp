#pragma once

#include "csDenseCellSet.hpp"

template <class NumericType, int D> class csSegmentCells {
  psSmartPointer<csDenseCellSet<NumericType, D>> cellSet = nullptr;
  std::string cellTypeString = "CellType";
  psMaterial bulkMaterial = psMaterial::GAS;

public:
  csSegmentCells(
      const psSmartPointer<csDenseCellSet<NumericType, D>> &passedCellSet)
      : cellSet(passedCellSet) {}

  csSegmentCells(
      const psSmartPointer<csDenseCellSet<NumericType, D>> &passedCellSet,
      const std::string cellTypeString, const psMaterial passedBulkMaterial)
      : cellSet(passedCellSet), cellTypeString(cellTypeString),
        bulkMaterial(passedBulkMaterial) {}

  void setCellSet(
      const psSmartPointer<csDenseCellSet<NumericType, D>> &passedCellSet) {
    cellSet = passedCellSet;
  }

  void setCellTypeString(const std::string passedCellTypeString) {
    cellTypeString = passedCellTypeString;
  }

  void setBulkMaterial(const psMaterial passedBulkMaterial) {
    bulkMaterial = passedBulkMaterial;
  }

  void apply() {
    auto cellType = cellSet->addScalarData(cellTypeString, -1.);
    cellSet->buildNeighborhood();
    auto materials = cellSet->getScalarData("Material");

#pragma omp parallel for
    for (unsigned i = 0; i < materials->size(); ++i) {
      if (!psMaterialMap::isMaterial(materials->at(i), bulkMaterial)) {
        auto neighbors = cellSet->getNeighbors(i);
        for (auto n : neighbors) {
          if (n >= 0 &&
              psMaterialMap::isMaterial(materials->at(n), bulkMaterial)) {
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
