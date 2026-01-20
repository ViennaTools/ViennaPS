#pragma once

#include "../psDomain.hpp"
#include "psProcessModel.hpp"
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

VIENNAPS_TEMPLATE_ND struct ProcessContext {
  // Core components
  SmartPointer<Domain<NumericType, D>> domain;
  SmartPointer<ProcessModelBase<NumericType, D>> model;

  // Process parameters
  double processDuration = 0.0;
  double processTime = 0.0;
  double timeStep = 0.0;

  // Configuration
  AdvectionParameters advectionParams;
  RayTracingParameters rayTracingParams;
  CoverageParameters coverageParams;
  AtomicLayerProcessParameters atomicLayerParams;
  std::string intermediateOutputPath = "";

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
    bool domainHasPeriodicBoundaries = false;
  } flags;

  void updateFlags() {
    assert(model && "Process model must be set before updating flags.");
    assert(domain && "Domain must be set before updating flags.");
    flags.isGeometric = model->getGeometricModel() != nullptr;
    flags.useFluxEngine = model->useFluxEngine();
    flags.useAdvectionCallback = model->getAdvectionCallback() != nullptr;
    flags.useProcessParams =
        model->getSurfaceModel() &&
        model->getSurfaceModel()->getProcessParameters() != nullptr;
    flags.isAnalytic = model->getVelocityField() && !model->useFluxEngine();
    flags.isALP = model->isALPModel();
    flags.useCoverages = flags.isALP || flags.useCoverages;

    const auto &grid = domain->getGrid();
    for (unsigned i = 0; i < D; ++i) {
      if (grid.getBoundaryConditions(i) ==
          viennals::BoundaryConditionEnum::PERIODIC_BOUNDARY) {
        flags.domainHasPeriodicBoundaries = true;
        break;
      }
    }
  }

  void printFlags() const {
    if (!Logger::hasDebug())
      return;

    std::stringstream stream;
    stream << "Process Context Flags:";
    stream << "\n\tisGeometric: " << util::boolString(flags.isGeometric)
           << "\n\tuseFluxEngine: " << util::boolString(flags.useFluxEngine)
           << "\n\tuseAdvectionCallback: "
           << util::boolString(flags.useAdvectionCallback)
           << "\n\tuseProcessParams: "
           << util::boolString(flags.useProcessParams)
           << "\n\tuseCoverages: " << util::boolString(flags.useCoverages)
           << "\n\tisALP: " << util::boolString(flags.isALP)
           << "\n\tisAnalytic: " << util::boolString(flags.isAnalytic)
           << "\n\tdomainHasPeriodicBoundaries: "
           << util::boolString(flags.domainHasPeriodicBoundaries);
    if (model) {
      stream << "\nProcess Name: "
             << model->getProcessName().value_or("default");
      stream << "\n\tHas GPU Model: " << util::boolString(model->hasGPUModel());
    }
    VIENNACORE_LOG_DEBUG(stream.str());
  }

  void resetTime() {
    processTime = 0.0;
    timeStep = 0.0;
    currentIteration = 0;
  }

  std::string getProcessName() const {
    return model->getProcessName().value_or("default");
  }

  bool needsExtendedVelocities() const {
    return advectionParams.spatialScheme ==
               SpatialScheme::LAX_FRIEDRICHS_1ST_ORDER ||
           advectionParams.spatialScheme ==
               SpatialScheme::LAX_FRIEDRICHS_2ND_ORDER ||
           advectionParams.spatialScheme ==
               SpatialScheme::LOCAL_LAX_FRIEDRICHS_1ST_ORDER ||
           advectionParams.spatialScheme ==
               SpatialScheme::LOCAL_LAX_FRIEDRICHS_2ND_ORDER ||
           advectionParams.spatialScheme ==
               SpatialScheme::LOCAL_LAX_FRIEDRICHS_ANALYTICAL_1ST_ORDER ||
           advectionParams.spatialScheme ==
               SpatialScheme::STENCIL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER ||
           advectionParams.temporalScheme ==
               TemporalScheme::RUNGE_KUTTA_2ND_ORDER ||
           advectionParams.temporalScheme ==
               TemporalScheme::RUNGE_KUTTA_3RD_ORDER;
  }
};

} // namespace viennaps
