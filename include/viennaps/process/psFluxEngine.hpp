#pragma once

#include "psProcessContext.hpp"

#include <vcTimer.hpp>

namespace viennaps {

VIENNAPS_TEMPLATE_ND(NumericType, D) class FluxEngine {
protected:
  viennacore::Timer<> timer_;
  unsigned fluxCalculationsCount_ = 0;

public:
  virtual ~FluxEngine() = default;

  // Implementation specific functions (to be implemented by derived classes,
  // currently CPU or GPU Process)
  virtual ProcessResult checkInput(ProcessContext<NumericType, D> &context) = 0;

  virtual ProcessResult initialize(ProcessContext<NumericType, D> &context) = 0;

  virtual ProcessResult
  updateSurface(ProcessContext<NumericType, D> &context) = 0;

  // Flux from source plane to surface
  virtual ProcessResult calculateSourceFluxes(
      ProcessContext<NumericType, D> &context,
      viennacore::SmartPointer<viennals::PointData<NumericType>> &fluxes) = 0;

  // Flux from surface to surface (e.g. desorption flux), add to source fluxes
  virtual ProcessResult calculateSurfaceFluxes(
      ProcessContext<NumericType, D> &context,
      viennacore::SmartPointer<viennals::PointData<NumericType>> &fluxes) = 0;

  auto &getTimer() const { return timer_; }
  void resetTimer() { timer_.reset(); }
  auto getFluxCalculationsCount() const { return fluxCalculationsCount_; }

  static void
  combineFluxes(viennals::PointData<NumericType> &fluxes,
                viennals::PointData<NumericType> &desorptionFluxes) {
    assert(fluxes.getScalarDataSize() == desorptionFluxes.getScalarDataSize());
#pragma omp parallel for
    for (int dataIdx = 0; dataIdx < fluxes.getScalarDataSize(); ++dataIdx) {
      auto fluxData = fluxes.getScalarData(dataIdx);
      auto desorptionData = desorptionFluxes.getScalarData(dataIdx);
      assert(fluxData);
      assert(desorptionData);
      assert(fluxData->size() == desorptionData->size());

      for (std::size_t i = 0; i < fluxData->size(); ++i) {
        (*fluxData)[i] += (*desorptionData)[i];
      }
    }
  }
};

} // namespace viennaps
