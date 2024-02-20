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
    cellSet->addScalarData("FluxReduce", 0.);
    cellSet->addScalarData("FluxAdsorbed", 0.);

    top = cellSet->getBoundingBox()[1][D - 1];
  }

  NumericType timeStep(const bool deposit = true) {
    auto &cellSet = domain->getCellSet();
    auto gridDelta = cellSet->getGridDelta();
    auto cellType = cellSet->getScalarData("CellType");
    auto flux = cellSet->getScalarData("Flux");
    auto adsorbed = cellSet->getScalarData("Adsorbed");
    auto fluxReduce = cellSet->getScalarData("FluxReduce");
    auto fluxAdsorbed = cellSet->getScalarData("FluxAdsorbed");
    std::fill(fluxReduce->begin(), fluxReduce->end(), 0.);

    const NumericType dt = std::min(
        gridDelta * gridDelta / diffusionCoefficient * timeStabilityFactor,
        NumericType(1.));
    const NumericType C = dt * diffusionCoefficient / (gridDelta * gridDelta);

    // shared
    std::vector<NumericType> newFlux(cellType->size(), 0.);

#pragma omp parallel shared(newFlux)
    {
      // local
      std::vector<NumericType> reduceFlux(cellType->size(), 0.);

#pragma omp for
      for (unsigned i = 0; i < cellType->size(); ++i) {
        const auto &neighbors = cellSet->getNeighbors(i);

        if (cellType->at(i) == 1.) {
          // inner cell
          auto center = cellSet->getCellCenter(i);
          if (center[D - 1] > top - gridDelta) {
            newFlux[i] = inFlux;
          } else {
            // Diffusion
            newFlux[i] = diffusion(flux, cellType, i, neighbors, C);
          }

        } else if (cellType->at(i) == 0. &&
                   adsorbed->at(i) < depositionThreshold) {
          // surface cell
          NumericType adsorbedAmount = 0.;
          int num_neighbors = 0;
          for (auto n : neighbors) {
            if (n >= 0 && cellType->at(n) == 1.) {
              auto adsorb = dt * adsorptionRate * flux->at(n);
              adsorbedAmount += adsorb;
              reduceFlux[n] += adsorb;
              num_neighbors++;
            }
          }
          if (num_neighbors > 1)
            adsorbedAmount /= num_neighbors;
          adsorbed->at(i) += adsorbedAmount;
          fluxAdsorbed->at(i) = adsorbedAmount;
          adsorbed->at(i) = std::min(adsorbed->at(i), depositionThreshold);
        }
      }

#pragma omp critical
      {
        for (unsigned i = 0; i < cellType->size(); ++i) {
          flux->at(i) -= reduceFlux[i];
          fluxReduce->at(i) += reduceFlux[i];
        }
      }
    }

    if (deposit) {
      for (unsigned i = 0; i < cellType->size(); ++i) {
        if (cellType->at(i) == 0. && adsorbed->at(i) > depositionThreshold) {
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
    }

#pragma omp parallel for
    for (unsigned i = 0; i < cellType->size(); ++i) {
      if (cellType->at(i) == 1.)
        flux->at(i) = newFlux[i];
    }

    return dt;
  }

private:
  static NumericType diffusion(const std::vector<NumericType> *flux,
                               const std::vector<NumericType> *cellType,
                               const unsigned i,
                               const std::array<int, 2 * D> &neighbors,
                               const NumericType C) {
    NumericType newFlux = 0.;
    int num_neighbors = 0;
    for (auto n : neighbors) {
      if (n >= 0 && cellType->at(n) == 1.) {
        newFlux += flux->at(n);
        num_neighbors++;
      }
    }
    return flux->at(i) + C * (newFlux - num_neighbors * flux->at(i));
  }

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