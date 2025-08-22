#pragma once

#include "psDomain.hpp"
#include "psProcessModelBase.hpp"
#include "psProcessParams.hpp"

namespace viennaps {

enum class ProcessResult {
  SUCCESS,
  INVALID_INPUT,
  EARLY_TERMINATION,
  CONVERGENCE_FAILURE,
  USER_INTERRUPTED,
  FAILURE,
  NOT_IMPLEMENTED
};

template <typename NumericType, int D> struct ProcessContext {
  // Core components
  SmartPointer<Domain<NumericType, D>> domain;
  SmartPointer<ProcessModelBase<NumericType, D>> model;

  // Process parameters
  NumericType processDuration = 0.0;
  NumericType processTime = 0.0;
  NumericType previousTimeStep = 0.0;

  // Configuration
  AdvectionParameters<NumericType> advectionParams;
  RayTracingParameters<NumericType, D> rayTracingParams;
  unsigned maxIterations = std::numeric_limits<unsigned>::max();
  NumericType coverageDeltaThreshold = 0.0;
  bool coveragesInitialized = false;

  // Simulation state
  unsigned currentIteration = 0;
  SmartPointer<viennals::Mesh<NumericType>> diskMesh;

  // Computed flags (derived from model state)
  struct Flags {
    bool useFluxEngine = false;
    bool useAdvectionCallback = false;
    bool useProcessParams = false;
    bool useCoverages = false;
    bool isGeometric = false;
  } flags;

  void updateFlags() {
    flags.isGeometric = model->getGeometricModel() != nullptr;
    flags.useFluxEngine = model->useFluxEngine();
    flags.useAdvectionCallback = model->getAdvectionCallback() != nullptr;
    flags.useProcessParams =
        model->getSurfaceModel() &&
        model->getSurfaceModel()->getProcessParameters() != nullptr;
  }

  void resetTime() {
    processTime = 0.0;
    previousTimeStep = 0.0;
  }

  std::string getProcessName() const {
    return model->getProcessName().value_or("default");
  }
};

} // namespace viennaps