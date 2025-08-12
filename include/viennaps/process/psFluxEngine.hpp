#pragma once

#include "psProcessContext.hpp"

namespace viennaps {

enum class FluxEngineType {
  CPU_Disk,
  GPU_Triangle,
  GPU_Disk,
  GPU_LevelSet
}

template <typename NumericType, int D>
class FluxEngine {
public:
  FluxEngine(ProcessContext<NumericType, D> &context) : context_(context) {
    // Initialize the flux engine with the provided context
  }

  virtual ~FluxEngine() = default;

  // Implementation specific functions (to be implemented by derived classes,
  // currently CPU or GPU Process)
  virtual bool checkInput() = 0;

  virtual void initialize() = 0;

  virtual void updateSurface() = 0;

  virtual viennacore::SmartPointer<viennals::PointData<NumericType>>
  calculateFluxes() = 0;

protected:
  ProcessContext<NumericType, D> &context_;
};

} // namespace viennaps