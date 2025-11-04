#pragma once

#include "psProcessContext.hpp"

#include <vcTimer.hpp>

namespace viennaps {

enum class FluxEngineType {
  // Platform, Surface representation
  AUTO,         // Automatic selection
  CPU_DISK,     // CPU, Disk-based
  CPU_TRIANGLE, // CPU, Triangle-based
  GPU_DISK,     // GPU, Disk-based
  GPU_TRIANGLE, // GPU, Triangle-based
  GPU_LINE,     // GPU, Line-based
  // GPU_LEVEL_SET // Future implementations
};

inline std::string to_string(FluxEngineType type) {
  switch (type) {
  case FluxEngineType::AUTO:
    return "AUTO";
  case FluxEngineType::CPU_DISK:
    return "CPU_DISK";
  case FluxEngineType::GPU_TRIANGLE:
    return "GPU_TRIANGLE";
  case FluxEngineType::GPU_DISK:
    return "GPU_DISK";
  case FluxEngineType::GPU_LINE:
    return "GPU_LINE";
  default:
    return "UNKNOWN";
  }
}

template <typename NumericType, int D> class FluxEngine {
protected:
  viennacore::Timer<> timer_;

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

  auto &getTimer() const { return timer_; }
  void resetTimer() { timer_.reset(); }
};

} // namespace viennaps