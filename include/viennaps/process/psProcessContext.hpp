#pragma once

#include "../psDomain.hpp"
#include "psProcessModelBase.hpp"
#include "psProcessParams.hpp"
#include "psTranslationField.hpp"

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
  double processDuration = 0.0;
  double processTime = 0.0;
  double timeStep = 0.0;

  // Configuration
  AdvectionParameters advectionParams;
  RayTracingParameters<D> rayTracingParams;
  CoverageParameters coverageParams;
  AtomicLayerProcessParameters atomicLayerParams;

  // Simulation state
  unsigned currentIteration = 0;
  SmartPointer<viennals::Mesh<NumericType>> diskMesh;
  SmartPointer<TranslationField<NumericType, D>> translationField;

  // Computed flags (derived from model state)
  struct Flags {
    bool useFluxEngine = false;
    bool useAdvectionCallback = false;
    bool useProcessParams = false;
    bool useCoverages = false;
    bool isALP = false;
    bool isAnalytic = false;
    bool isGeometric = false;
  } flags;

  void updateFlags() {
    flags.isGeometric = model->getGeometricModel() != nullptr;
    flags.useFluxEngine = model->useFluxEngine();
    flags.useAdvectionCallback = model->getAdvectionCallback() != nullptr;
    flags.useProcessParams =
        model->getSurfaceModel() &&
        model->getSurfaceModel()->getProcessParameters() != nullptr;
    flags.isAnalytic =
        model->getVelocityField() &&
        model->getVelocityField()->getTranslationFieldOptions() == 0;
    flags.isALP = model->isALPModel();
    flags.useCoverages = flags.isALP || flags.useCoverages;
  }

  void resetTime() {
    processTime = 0.0;
    timeStep = 0.0;
    currentIteration = 0;
  }

  std::string getProcessName() const {
    return model->getProcessName().value_or("default");
  }
};

} // namespace viennaps