#pragma once

#include "psProcessContext.hpp"

#include <vcTimer.hpp>

namespace viennaps {

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