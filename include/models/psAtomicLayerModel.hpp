#pragma once

#include <psDomain.hpp>

template <class NumericType, int D> class psAtomicLayerModel {
  psSmartPointer<psDomain<NumericType, D>> domain = nullptr;
  NumericType inFlux = 1.0;
  NumericType top = 0.;
  const NumericType diffusionCoefficient = 20.0;
  const NumericType timeStabilityFactor = 0.245;
  const NumericType adsorptionRate = 0.1;
  const NumericType depositionThreshold = 10.;

public:
  psAtomicLayerModel(
      const psSmartPointer<psDomain<NumericType, D>> &passedDomain,
      NumericType passedDiffusionCoefficient, NumericType passedInFlux,
      NumericType passedAdsorptionRate, NumericType passedDepositionThreshold)
      : diffusionCoefficient(passedDiffusionCoefficient), inFlux(passedInFlux),
        adsorptionRate(passedAdsorptionRate),
        depositionThreshold(passedDepositionThreshold) {
    domain = passedDomain;
    auto &cellSet = domain->getCellSet();
    segmentCells();
    cellSet->addScalarData("Flux", 0.);
    cellSet->addScalarData("Adsorbed", 0.);
    top = cellSet->getBoundingBox()[1][D - 1];
  }

  NumericType timeStep() {
    auto &cellSet = domain->getCellSet();
    auto gridDelta = cellSet->getGridDelta();
    auto cellType = cellSet->getScalarData("CellType");
    auto flux = cellSet->getScalarData("Flux");
    auto adsorbed = cellSet->getScalarData("Adsorbed");

    const NumericType dt = std::min(
        gridDelta * gridDelta / diffusionCoefficient * timeStabilityFactor,
        NumericType(1.));
    const NumericType C = dt * diffusionCoefficient / (gridDelta * gridDelta);

    std::vector<NumericType> newFlux(cellType->size(), 0.);

#pragma omp parallel for
    for (unsigned i = 0; i < cellType->size(); ++i) {
      const auto &neighbors = cellSet->getNeighbors(i);

      if (cellType->at(i) == 1.) {
        // inner cell
        auto center = cellSet->getCellCenter(i);
        if (center[D - 1] > top - gridDelta) {
          newFlux[i] = inFlux;
        } else {
          // Diffusion
          int num_neighbors = 0;
          for (auto n : neighbors) {
            if (n >= 0 && cellType->at(n) == 1.) {
              newFlux[i] += flux->at(n);
              num_neighbors++;
            }
          }
          newFlux[i] =
              flux->at(i) + C * (newFlux[i] - num_neighbors * flux->at(i));
        }

      } else if (cellType->at(i) == 0.) {
        // surface cell
        int num_neighbors = 0;
        NumericType adsorbedAmount = 0.;
        for (auto n : neighbors) {
          if (n >= 0 && cellType->at(n) == 1.) {
            adsorbedAmount += flux->at(n);
            num_neighbors++;
          }
        }
        if (num_neighbors > 1)
          adsorbedAmount /= num_neighbors;
        adsorbed->at(i) += dt * adsorptionRate * adsorbedAmount;
      }
    }

    for (unsigned i = 0; i < cellType->size(); ++i) {
      if (cellType->at(i) == 0.) {
        if (adsorbed->at(i) > depositionThreshold) {
          const auto &neighbors = cellSet->getNeighbors(i);
          for (auto n : neighbors) {
            if (n >= 0 && cellType->at(n) == 1.) {
              cellType->at(n) = 0.;
              adsorbed->at(n) = 0.;
              flux->at(n) = 0.;
            }
          }
          adsorbed->at(i) = 0.;
          flux->at(i) = 0.;
          cellType->at(i) = -1.;
        }
      }

      if (cellType->at(i) == 1.)
        flux->at(i) = newFlux[i];
    }

    return dt;
  }

private:
  void segmentCells() {
    auto &cellSet = domain->getCellSet();
    auto cellType = cellSet->addScalarData("CellType", -1.);

    cellSet->buildNeighborhood();
    auto materials = cellSet->getScalarData("Material");

    for (unsigned i = 0; i < materials->size(); ++i) {
      if (!psMaterialMap::isMaterial(materials->at(i), psMaterial::GAS)) {
        auto neighbors = cellSet->getNeighbors(i);
        for (auto n : neighbors) {
          if (n >= 0 &&
              psMaterialMap::isMaterial(materials->at(n), psMaterial::GAS)) {
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