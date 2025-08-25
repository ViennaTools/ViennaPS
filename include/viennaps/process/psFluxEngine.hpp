#pragma once

#include "psProcessContext.hpp"

namespace viennaps {

enum class FluxEngineType {
  // Platform, Surface representation
  CPU_DISK,     // CPU, Disk-based
  GPU_TRIANGLE, // GPU, Triangle-based
  // GPU_DISK,   // GPU, Disk-based
  // GPU_LEVEL_SET // Future implementations
};

template <typename NumericType, int D> class FluxEngine {
public:
  virtual ~FluxEngine() = default;

  // Implementation specific functions (to be implemented by derived classes,
  // currently CPU or GPU Process)
  virtual ProcessResult checkInput(ProcessContext<NumericType, D> &context) = 0;

  virtual ProcessResult initialize(ProcessContext<NumericType, D> &context) = 0;

  virtual ProcessResult
  updateSurface(ProcessContext<NumericType, D> &context) = 0;

  virtual ProcessResult calculateFluxes(
      ProcessContext<NumericType, D> &context,
      viennacore::SmartPointer<viennals::PointData<NumericType>> &fluxes) = 0;
};

} // namespace viennaps