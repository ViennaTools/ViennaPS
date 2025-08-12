#pragma once

#include "psDomain.hpp"
#include "psProcessModelBase.hpp"
#include "psProcessParams.hpp"

namespace viennaps {

template <typename NumericType, int D> struct ProcessContext {
  // Core components
  SmartPointer<Domain<NumericType, D>> domain;
  SmartPointer<ProcessModelBase<NumericType, D>> model;

  // Process parameters
  NumericType processDuration = 0.0;
  NumericType processTime = 0.0;
  NumericType remainingTime = 0.0;

  // Configuration
  AdvectionParameters<NumericType> advectionParams;
  RayTracingParameters<NumericType, D> rayTracingParams;
  unsigned maxIterations = std::numeric_limits<unsigned>::max();
  NumericType coverageDeltaThreshold = 0.0;
  bool coveragesInitialized = false;

  // Runtime state
  SmartPointer<viennals::Mesh<NumericType>> diskMesh;
  SmartPointer<TranslationField<NumericType, D>> translationField;
  std::unique_ptr<std::ofstream> covMetricFile;

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

  std::string getProcessName() const {
    return model->getProcessName().value_or("default");
  }
};

} // namespace viennaps