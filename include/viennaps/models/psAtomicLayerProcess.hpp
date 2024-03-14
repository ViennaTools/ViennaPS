#pragma once

#include "psDomain.hpp"
#include "psLogger.hpp"

#include <random>

template <class NumericType, int D> class psAtomicLayerProcess {

public:
  struct Precursor {
    std::string name;
    NumericType meanThermalVelocity = 0.;
    NumericType adsorptionRate = 0.;
    NumericType desorptionRate = 0.;
    NumericType duration = 0.;
    NumericType inFlux = 1.;
  };

private:
  std::array<Precursor, 2> precursors;
  NumericType purge_meanThermalVelocity = 0.;
  NumericType purge_duration = 0.;

  NumericType reactionOrder = 1.;

  NumericType maxLambda = 0.;
  NumericType stabilityFactor = 0.245;
  NumericType maxTimeStep = 1.;
  NumericType printInterval = -1.;

public:
  psAtomicLayerProcess(
      const psSmartPointer<psDomain<NumericType, D>> &passedDomain,
      const bool passedEtch = false)
      : domain(passedDomain), etch(passedEtch),
        top(domain->getCellSet()->getBoundingBox()[1][D - 1]) {
    // initialize random number generator
    std::random_device rd;
    rng.seed(rd());
  }

  void setFirstPrecursor(std::string name, NumericType meanThermalVelocity,
                         NumericType adsorptionRate, NumericType desorptionRate,
                         NumericType duration, NumericType inFlux = 1.) {
    precursors[0].name = name;
    precursors[0].meanThermalVelocity = meanThermalVelocity;
    precursors[0].adsorptionRate = adsorptionRate;
    precursors[0].desorptionRate = desorptionRate;
    precursors[0].duration = duration;
    precursors[0].inFlux = inFlux;
  }

  void setFirstPrecursor(const Precursor &p) { precursors[0] = p; }

  void setSecondPrecursor(std::string name, NumericType meanThermalVelocity,
                          NumericType adsorptionRate,
                          NumericType desorptionRate, NumericType duration,
                          NumericType inFlux = 1.) {
    precursors[1].name = name;
    precursors[1].meanThermalVelocity = meanThermalVelocity;
    precursors[1].adsorptionRate = adsorptionRate;
    precursors[1].desorptionRate = desorptionRate;
    precursors[1].duration = duration;
    precursors[1].inFlux = inFlux;
  }

  void setSecondPrecursor(const Precursor &p) { precursors[1] = p; }

  void setPurgeParameters(NumericType meanThermalVelocity,
                          NumericType duration) {
    purge_meanThermalVelocity = meanThermalVelocity;
    purge_duration = duration;
  }

  void setReactionOrder(const NumericType &value) { reactionOrder = value; }

  void setMaxLambda(const NumericType &value) { maxLambda = value; }

  void setStabilityFactor(const NumericType &value) { stabilityFactor = value; }

  void setMaxTimeStep(const NumericType &value) { maxTimeStep = value; }

  void setPrintInterval(const NumericType &value) { printInterval = value; }

  void apply() {
    auto &cellSet = domain->getCellSet();
    flux = cellSet->addScalarData("Flux", 0.);
    depoProb = cellSet->addScalarData("ReactionProbability", 0.);
    cellSet->addScalarData(precursors[0].name, 0.);
    cellSet->addScalarData(precursors[1].name, 0.);

    lambda = cellSet->getScalarData("MeanFreePath");
    if (lambda == nullptr) {
      psLogger::getInstance()
          .addError("MeanFreePath scalar data not found.")
          .print();
    }

    cellType = cellSet->getScalarData("CellType");
    if (cellType == nullptr) {
      psLogger::getInstance()
          .addError("CellType scalar data not found.")
          .print();
    }

    std::string fileName =
        "ALP_" + precursors[0].name + "_" + precursors[1].name + "_";

    // P1 step
    double time = 0.;
    int i = 0, ii = 0;
    auto coverage = cellSet->getScalarData(precursors[0].name);
    auto adsorbat = cellSet->getScalarData(precursors[1].name);
    while (time < precursors[0].duration) {
      time +=
          timeStep(coverage, adsorbat, precursors[0].meanThermalVelocity,
                   precursors[0].adsorptionRate, precursors[0].desorptionRate,
                   precursors[0].inFlux, false);
      if (time - ii * 0.1 > 0) {
        ++ii;
        psLogger::getInstance()
            .addInfo("P1 step remaining time: " +
                     std::to_string(precursors[0].duration - time))
            .print();
      }
      if (time - i * printInterval > 0. && psLogger::getLogLevel() > 3) {
        cellSet->writeVTU(fileName + std::to_string(i++) + ".vtu");
      }
    }

    // Purge step
    time = 0.;
    int j = 0, jj = 0;
    while (time < purge_duration) {
      time += timeStep(coverage, adsorbat, purge_meanThermalVelocity,
                       precursors[0].adsorptionRate,
                       precursors[0].desorptionRate, 0., false);
      if (time - jj * 0.1 > 0) {
        ++jj;
        psLogger::getInstance()
            .addInfo("Purge step remaining time: " +
                     std::to_string(purge_duration - time))
            .print();
      }
      if (time - j * printInterval > 0. && psLogger::getLogLevel() > 3) {
        cellSet->writeVTU(fileName + std::to_string(i++) + ".vtu");
        ++j;
      }
    }

    auto flux = cellSet->getScalarData("Flux");
    std::fill(flux->begin(), flux->end(), 0.);

    // P2 step
    time = 0.;
    int k = 0, kk = 0;
    coverage = cellSet->getScalarData(precursors[1].name);
    adsorbat = cellSet->getScalarData(precursors[0].name);
    while (time < precursors[1].duration) {
      time +=
          timeStep(coverage, adsorbat, precursors[1].meanThermalVelocity,
                   precursors[1].adsorptionRate, precursors[1].desorptionRate,
                   precursors[1].inFlux, true);
      if (time - kk * 0.1 > 0) {
        ++kk;
        psLogger::getInstance()
            .addInfo("P2 step remaining time: " +
                     std::to_string(precursors[1].duration - time))
            .print();
      }
      if (time - k * printInterval > 0. && psLogger::getLogLevel() > 3) {
        cellSet->writeVTU(fileName + std::to_string(i++) + ".vtu");
        ++k;
      }
    }
  }

private:
  NumericType timeStep(std::vector<NumericType> *coverage,
                       std::vector<NumericType> *adsorbat,
                       const NumericType meanThermalVelocity,
                       const NumericType adsorptionRate,
                       const NumericType desorptionRate,
                       const NumericType inFlux, const bool modify) {
#ifdef VIENNAPS_PYTHON_BUILD
    if (PyErr_CheckSignals() != 0)
      throw pybind11::error_already_set();
#endif
    auto &cellSet = domain->getCellSet();

    const auto gridDelta = cellSet->getGridDelta();
    const NumericType diffusionFactor =
        meanThermalVelocity / 3.; // D = 1/3 * v * lambda
    const NumericType dt = std::min(
        gridDelta * gridDelta / (maxLambda * diffusionFactor) * stabilityFactor,
        maxTimeStep);
    // The time step has to fulfill the stability condition for the explicit
    // finite difference method, the stability factor has to smaller than 0.5.
    // https://en.wikipedia.org/wiki/Numerical_solution_of_the_convection%E2%80%93diffusion_equation#Stability_criteria
    const NumericType dtdx2 = dt / (gridDelta * gridDelta);

    // shared
    const unsigned numThreads = omp_get_max_threads();
    std::vector<NumericType> newFlux(cellType->size(), 0.);
    std::vector<std::vector<NumericType>> reduceFluxes(numThreads);

#pragma omp parallel
    {
      // local
      auto &reduceFlux = reduceFluxes[omp_get_thread_num()];
      reduceFlux.resize(cellType->size(), 0.);

#pragma omp for
      for (unsigned i = 0; i < cellType->size(); ++i) {
        const auto &neighbors = cellSet->getNeighbors(i);

        if (cellType->at(i) == 1.) {
          /* ----- Inner cell ----- */
          if (cellSet->getCellCenter(i)[D - 1] > top - gridDelta) {
            // Inlet/outlet at the top
            newFlux[i] = inFlux;
          } else {
            // Diffusion
            NumericType diffusion = 0.;
            for (const auto &n : neighbors) {
              if (n >= 0 && cellType->at(n) == 1.) {
                // harmonic mean
                auto interfaceDiffusivity =
                    2. / (1. / (lambda->at(i) * diffusionFactor) +
                          1. / (lambda->at(n) * diffusionFactor));
                diffusion += interfaceDiffusivity * (flux->at(n) - flux->at(i));
              }
            }
            newFlux[i] = flux->at(i) + dtdx2 * diffusion;
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
      for (unsigned i = 0; i < flux->size(); ++i) {
        if (cellType->at(i) == 1.) {
          NumericType reduce = 0.;
          for (unsigned j = 0; j < numThreads; j++) {
            reduce += reduceFluxes[j][i];
          }
          flux->at(i) = newFlux[i] - reduce;
          assert(flux->at(i) >= 0. && "Negative flux");
        }
      }
    } // end of parallel region

    if (modify) {
      std::uniform_real_distribution<NumericType> dist(0., 1.);
      for (unsigned i = 0; i < cellType->size(); ++i) {
        if (cellType->at(i) == 0.) {
          depoProb->at(i) =
              std::pow(coverage->at(i) * adsorbat->at(i), reactionOrder);
          if (dist(rng) < depoProb->at(i) * dt) {
            if (etch) {
              cellType->at(i) = 1.;
              coverage->at(i) = 0.;
              flux->at(i) = 0.;
              adsorbat->at(i) = 0.;
            } else {
              const auto &neighbors = cellSet->getNeighbors(i);
              for (auto n : neighbors) {
                if (n >= 0 && cellType->at(n) == 1.) {
                  cellType->at(n) = 2.;
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
        }
      }
    }

    return dt;
  }

private:
  psSmartPointer<psDomain<NumericType, D>> domain = nullptr;
  const bool etch = false;
  const NumericType top = 0.;
  std::mt19937_64 rng;
  std::vector<NumericType> *flux;
  std::vector<NumericType> *lambda;
  std::vector<NumericType> *cellType;
  std::vector<NumericType> *depoProb;
};
