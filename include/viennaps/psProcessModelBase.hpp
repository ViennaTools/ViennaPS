#pragma once

#include "psAdvectionCallback.hpp"
#include "psDomain.hpp"
#include "psGeometricModel.hpp"
#include "psSurfaceModel.hpp"
#include "psVelocityField.hpp"

namespace viennaps {

using namespace viennacore;

/// The process model combines all models (particle types, surface model,
/// geometric model, advection callback)
template <typename NumericType, int D> class ProcessModelBase {
protected:
  SmartPointer<SurfaceModel<NumericType>> surfaceModel = nullptr;
  SmartPointer<AdvectionCallback<NumericType, D>> advectionCallback = nullptr;
  SmartPointer<GeometricModel<NumericType, D>> geometricModel = nullptr;
  SmartPointer<VelocityField<NumericType, D>> velocityField = nullptr;
  std::optional<std::string> processName = std::nullopt;

public:
  virtual ~ProcessModelBase() = default;

  virtual void initialize(SmartPointer<Domain<NumericType, D>> domain,
                          const NumericType processDuration) {}
  virtual void reset() {}
  virtual bool useFluxEngine() { return false; }

  auto getSurfaceModel() const { return surfaceModel; }
  auto getAdvectionCallback() const { return advectionCallback; }
  auto getGeometricModel() const { return geometricModel; }
  auto getVelocityField() const { return velocityField; }
  auto getProcessName() const { return processName; }

  void setProcessName(const std::string &name) { processName = name; }

  void
  setSurfaceModel(SmartPointer<SurfaceModel<NumericType>> passedSurfaceModel) {
    surfaceModel = passedSurfaceModel;
  }

  void setAdvectionCallback(
      SmartPointer<AdvectionCallback<NumericType, D>> passedAdvectionCallback) {
    advectionCallback = passedAdvectionCallback;
  }

  void setGeometricModel(
      SmartPointer<GeometricModel<NumericType, D>> passedGeometricModel) {
    geometricModel = passedGeometricModel;
  }

  void setVelocityField(
      SmartPointer<VelocityField<NumericType, D>> passedVelocityField) {
    velocityField = passedVelocityField;
  }
};

} // namespace viennaps
