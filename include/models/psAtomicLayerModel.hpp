#pragma once

#include <psDomain.hpp>

#include <random>

template <class NumericType, int D> class psAtomicLayerModel {
  psSmartPointer<psDomain<NumericType, D>> domain = nullptr;
  NumericType top = 0.;
  NumericType maxDiffusivity = -1.;
  const NumericType stabilityFactor = 0.245;
  const std::string precursor_p1;
  const std::string precursor_p2;
  NumericType depositTime = 0.;
  int depositCount = 0;

  std::mt19937_64 rng;

public:
  psAtomicLayerModel(
      const psSmartPointer<psDomain<NumericType, D>> &passedDomain,
      std::string pPrecursor_p1, std::string pPrecursor_p2)
      : domain(passedDomain), precursor_p1(pPrecursor_p1),
        precursor_p2(pPrecursor_p2) {

    auto &cellSet = domain->getCellSet();
    segmentCells();
    cellSet->addScalarData("Flux", 0.);
    cellSet->addScalarData(precursor_p1, 0.);
    cellSet->addScalarData(precursor_p2, 0.);

    top = cellSet->getBoundingBox()[1][D - 1];

    // initialize RNG
    std::random_device rd;
    rng.seed(rd());
  }

  NumericType timeStep(const NumericType diffusionCoefficient,
                       const NumericType adsorptionRate,
                       const NumericType desorptionRate,
                       const NumericType inFlux, bool deposit) {
    auto &cellSet = domain->getCellSet();
    const auto gridDelta = cellSet->getGridDelta();
    const auto cellType = cellSet->getScalarData("CellType");
    auto flux = cellSet->getScalarData("Flux");
    std::string coverageName = deposit ? precursor_p2 : precursor_p1;
    auto coverage = cellSet->getScalarData(coverageName);
    auto adsorbat = cellSet->getScalarData(precursor_p1);
    if (coverage == nullptr) {
      std::cerr << "Coverage scalar data not found" << std::endl;
      exit(1);
    }
    if (adsorbat == nullptr) {
      std::cerr << "Adsorbat scalar data not found" << std::endl;
      exit(1);
    }

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

          if (cellSet->getCellCenter(i)[D - 1] > top - gridDelta) {
            // Inlet/outlet at the top
            newFlux[i] = inFlux;
          } else {
            // Diffusion
            newFlux[i] = diffusion(flux, cellType, i, neighbors, C);
          }
        } else if (cellType->at(i) == 0.) {
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
      depositTime += dt;
      if (depositTime - depositCount > 0) {
        std::uniform_real_distribution<NumericType> dist(0., 1.);
        for (unsigned i = 0; i < cellType->size(); ++i) {
          if (dist(rng) < coverage->at(i) && dist(rng) < adsorbat->at(i)) {
            const auto &neighbors = cellSet->getNeighbors(i);
            for (auto n : neighbors) {
              if (n >= 0 && cellType->at(n) == 1.) {
                cellType->at(n) = 0.;
                coverage->at(n) = 0.;
                flux->at(n) = 0.;
                adsorbat->at(n) = 0.;
              }
            }
            coverage->at(i) = 0.;
            flux->at(i) = 0.;
            cellType->at(i) = -1.;
            adsorbat->at(i) = 0.;
          }
        }
        depositCount++;
      }
    }

    return dt;
  }

private:
  static NumericType diffusion(const std::vector<NumericType> *flux,
                               const std::vector<NumericType> *cellType,
                               const unsigned i,
                               const std::array<int, 2 * D> &neighbors,
                               const NumericType Dij) {
    NumericType newFlux = 0.;
    int num_neighbors = 0;
    for (const auto &n : neighbors) {
      if (n >= 0 && cellType->at(n) == 1.) {
        newFlux += flux->at(n);
        num_neighbors++;
      }
    }
    return Dij * (newFlux - num_neighbors * flux->at(i));
  }

  void segmentCells() {}
};