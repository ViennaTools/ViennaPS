#pragma once

#include <psDomain.hpp>
#include <psLogger.hpp>

#include <random>

template <class NumericType, int D, bool isDConst = true> class psDiffusion {
  NumericType diffusionCoefficient = 0.;
  NumericType meanThermalVelocity = 0.;
  NumericType maxLambda = 0.;
  NumericType inFlux = 0.;
  NumericType stabilityFactor = 0.245;
  NumericType duration = 1.;
  NumericType printInterval = -1.;
  NumericType maxTimeStep = 1.;

  psMaterial material = psMaterial::GAS;

public:
  psDiffusion(const psSmartPointer<psDomain<NumericType, D>> &passedDomain)
      : domain(passedDomain),
        top(domain->getCellSet()->getBoundingBox()[1][D - 1]) {}

  void setDiffusionCoefficient(const NumericType &value) {
    diffusionCoefficient = value;
  }

  void setMeanThermalVelocity(const NumericType &value) {
    meanThermalVelocity = value;
  }

  void setMaxLambda(const NumericType &value) { maxLambda = value; }

  void setInFlux(const NumericType &value) { inFlux = value; }

  void setStabilityFactor(const NumericType &value) { stabilityFactor = value; }

  void setDuration(const NumericType &value) { duration = value; }

  void setPrintInterval(const NumericType &value) { printInterval = value; }

  void setMaxTimeStep(const NumericType &value) { maxTimeStep = value; }

  void apply() {
    auto &cellSet = domain->getCellSet();
    cellSet->buildNeighborhood();
    flux = cellSet->addScalarData("Flux", 0.);

    if constexpr (!isDConst) {
      lambda = cellSet->getScalarData("MeanFreePath");
      if (lambda == nullptr) {
        psLogger::getInstance()
            .addError("MeanFreePath scalar data not found.")
            .print();
      }
    }

    double time = 0.;
    int i = 0;
    while (time < duration) {
      time += timeStep();
      if (time - i * printInterval > 0. && psLogger::getLogLevel() > 3) {
        psLogger::getInstance()
            .addInfo("Diffusion remaining time: " +
                     std::to_string(duration - time))
            .print();
        cellSet->writeVTU("diffusion_" + std::to_string(i++) + ".vtu");
      }
    }
  }

private:
  NumericType timeStep() {
#ifdef VIENNAPS_PYTHON_BUILD
    if (PyErr_CheckSignals() != 0)
      throw pybind11::error_already_set();
#endif
    auto &cellSet = domain->getCellSet();
    auto materials = cellSet->getScalarData("Material");

    const auto gridDelta = cellSet->getGridDelta();
    const NumericType diffusionFactor =
        meanThermalVelocity / 3.; // D = 1/3 * v * lambda
    NumericType timeStep;
    if constexpr (!isDConst) {
      timeStep = gridDelta * gridDelta / (maxLambda * diffusionFactor) *
                 stabilityFactor;
    } else {
      timeStep = gridDelta * gridDelta / diffusionCoefficient * stabilityFactor;
    }

    const NumericType dt = std::min(timeStep, maxTimeStep);
    // The time step has to fulfill the stability condition for the explicit
    // finite difference method, the stability factor has to smaller than 0.5.
    // https://en.wikipedia.org/wiki/Numerical_solution_of_the_convection%E2%80%93diffusion_equation#Stability_criteria
    const NumericType dtdx2 = dt / (gridDelta * gridDelta);
    const NumericType C = stabilityFactor;

    // shared
    const unsigned numThreads = omp_get_max_threads();
    std::vector<NumericType> newFlux(flux->size(), 0.);

#pragma omp parallel for
    for (unsigned i = 0; i < flux->size(); ++i) {
      if (!psMaterialMap::isMaterial(materials->at(i), material))
        continue;

      const auto &neighbors = cellSet->getNeighbors(i);
      if (cellSet->getCellCenter(i)[D - 1] > top - gridDelta) {
        // Inlet/outlet at the top
        newFlux[i] = inFlux;
      } else {
        // Diffusion
        if constexpr (isDConst) {
          int numNeighbors = 0;
          for (const auto &n : neighbors) {
            if (n >= 0 &&
                psMaterialMap::isMaterial(materials->at(n), material)) {
              newFlux[i] += flux->at(n);
              numNeighbors++;
            }
          }
          newFlux[i] =
              flux->at(i) +
              C * (newFlux[i] -
                   static_cast<NumericType>(numNeighbors) * flux->at(i));
        } else {
          NumericType diffusion = 0.;
          for (const auto &n : neighbors) {
            if (n >= 0 &&
                psMaterialMap::isMaterial(materials->at(n), material)) {
              // harmonic mean
              auto interfaceDiffusivity =
                  2. / (1. / (lambda->at(i) * diffusionFactor) +
                        1. / (lambda->at(n) * diffusionFactor));
              diffusion += interfaceDiffusivity * (flux->at(n) - flux->at(i));
            }
          }
          newFlux[i] = flux->at(i) + dtdx2 * diffusion;
        }
      }
    }

#pragma omp parallel for
    for (unsigned i = 0; i < flux->size(); ++i) {
      flux->at(i) = newFlux[i];
    }

    return dt;
  }

private:
  psSmartPointer<psDomain<NumericType, D>> domain = nullptr;
  const NumericType top = 0.;
  std::vector<NumericType> *flux;
  std::vector<NumericType> *lambda;
};