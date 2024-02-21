#pragma once

#include <psDomain.hpp>

template <class NumericType, int D> class psAtomicLayerModel {
  psSmartPointer<psDomain<NumericType, D>> domain = nullptr;
  NumericType top = 0.;
  const NumericType inFlux = 1.0;
  const NumericType diffusionCoefficient = 20.0;
  const NumericType adsorptionRate = 0.1;
  const NumericType depositionThreshold = 1.;
  const NumericType desorptionRate = 0.1;
  const NumericType stabilityFactor = 0.245;

public:
  psAtomicLayerModel(
      const psSmartPointer<psDomain<NumericType, D>> &passedDomain,
      NumericType passedDiffusionCoefficient, NumericType passedInFlux,
      NumericType passedAdsorptionRate, NumericType passedDesorptionRate,
      NumericType passedDepositionThreshold)
      : domain(passedDomain), diffusionCoefficient(passedDiffusionCoefficient),
        inFlux(passedInFlux), adsorptionRate(passedAdsorptionRate),
        desorptionRate(passedDesorptionRate),
        depositionThreshold(passedDepositionThreshold) {

    auto &cellSet = domain->getCellSet();
    segmentCells();
    cellSet->addScalarData("Flux", 0.);
    cellSet->addScalarData("SurfaceCoverage", 0.);

    top = cellSet->getBoundingBox()[1][D - 1];
  }

  NumericType timeStep(const bool deposit = true) {
    auto &cellSet = domain->getCellSet();
    auto gridDelta = cellSet->getGridDelta();
    auto cellType = cellSet->getScalarData("CellType");
    auto flux = cellSet->getScalarData("Flux");
    auto coverage = cellSet->getScalarData("SurfaceCoverage");

    const NumericType dt =
        std::min(gridDelta * gridDelta / diffusionCoefficient * stabilityFactor,
                 NumericType(1.));
    // The time step has to fulfill the stability condition for the explicit
    // finite difference method, the stability factor has to smaller than 0.5.
    // https://en.wikipedia.org/wiki/Numerical_solution_of_the_convection%E2%80%93diffusion_equation#Stability_criteria
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
          /* ----- GAS cell ----- */

          const auto &center = cellSet->getCellCenter(i);
          if (center[D - 1] > top - gridDelta) {
            // Inlet at the top
            newFlux[i] = inFlux;
          } else {
            // Diffusion
            newFlux[i] = diffusion(flux, cellType, i, neighbors, C);
          }
        } else if (cellType->at(i) == 0. &&
                   coverage->at(i) < depositionThreshold) {
          /* ----- Surface cell ----- */

          // Adsorption
          NumericType effectiveSticking =
              adsorptionRate * (1 - coverage->at(i));
          NumericType adsorbedAmount = 0.;
          int num_neighbors = 0;
          for (auto n : neighbors) {
            if (n >= 0 && cellType->at(n) == 1.) {
              auto adsorb = dt * effectiveSticking * flux->at(n);
              adsorbedAmount += adsorb;
              reduceFlux[n] += adsorb;
              num_neighbors++;
            }
          }
          coverage->at(i) += num_neighbors > 1 ? adsorbedAmount / num_neighbors
                                               : adsorbedAmount;

          // Desorption
          NumericType desorbedAmount = dt * desorptionRate * coverage->at(i);
          coverage->at(i) -= desorbedAmount;
          for (const auto &n : neighbors) {
            if (n >= 0 && cellType->at(n) == 1.) {
              reduceFlux[n] -= desorbedAmount / num_neighbors;
            }
          }
        }
      }

#pragma omp barrier
#pragma omp for
      for (unsigned i = 0; i < cellType->size(); ++i) {
        if (cellType->at(i) == 1.)
          flux->at(i) = newFlux[i];
      }

#pragma omp barrier
#pragma omp critical
      {
        // add/subtract desorbed/adsorbed flux
        for (unsigned i = 0; i < cellType->size(); ++i) {
          if (cellType->at(i) == 1. && reduceFlux[i] != 0.)
            flux->at(i) -= reduceFlux[i];
        }
      }
    } // end of parallel region

    if (deposit) {
      for (unsigned i = 0; i < cellType->size(); ++i) {
        if (cellType->at(i) == 0. && coverage->at(i) > depositionThreshold) {
          const auto &neighbors = cellSet->getNeighbors(i);
          for (auto n : neighbors) {
            if (n >= 0 && cellType->at(n) == 1.) {
              cellType->at(n) = 0.;
              coverage->at(n) = 0.;
              flux->at(n) = 0.;
            }
          }
          coverage->at(i) = 0.;
          flux->at(i) = 0.;
          cellType->at(i) = -1.;
        }
      }
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
    for (const auto &n : neighbors) {
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