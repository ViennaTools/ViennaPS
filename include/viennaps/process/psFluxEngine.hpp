#pragma once

#include "psProcessContext.hpp"

namespace viennaps {

enum class FluxEngineType { CPU_Disk, GPU_Triangle, GPU_Disk, GPU_LevelSet };

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