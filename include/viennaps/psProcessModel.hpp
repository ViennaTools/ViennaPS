#pragma once

#include "psAdvectionCallback.hpp"
#include "psGeometricModel.hpp"
#include "psSmartPointer.hpp"
#include "psSurfaceModel.hpp"
#include "psVelocityField.hpp"

#include <rayParticle.hpp>

/// The process model combines all models (particle types, surface model,
/// geometric model, advection callback)
template <typename NumericType, int D> class psProcessModel {
protected:
  using ParticleTypeList =
      std::vector<std::unique_ptr<rayAbstractParticle<NumericType>>>;

  psSmartPointer<ParticleTypeList> particles = nullptr;
  std::vector<int> particleLogSize;
  psSmartPointer<psSurfaceModel<NumericType>> surfaceModel = nullptr;
  psSmartPointer<psAdvectionCallback<NumericType, D>> advectionCallback =
      nullptr;
  psSmartPointer<psGeometricModel<NumericType, D>> geometricModel = nullptr;
  psSmartPointer<psVelocityField<NumericType>> velocityField = nullptr;
  std::optional<std::string> processName = std::nullopt;
  std::optional<std::array<NumericType, 3>> primaryDirection = std::nullopt;

public:
  virtual psSmartPointer<ParticleTypeList> getParticleTypes() const {
    return particles;
  }
  virtual psSmartPointer<psSurfaceModel<NumericType>> getSurfaceModel() const {
    return surfaceModel;
  }
  virtual psSmartPointer<psAdvectionCallback<NumericType, D>>
  getAdvectionCallback() const {
    return advectionCallback;
  }
  virtual psSmartPointer<psGeometricModel<NumericType, D>>
  getGeometricModel() const {
    return geometricModel;
  }
  virtual psSmartPointer<psVelocityField<NumericType>>
  getVelocityField() const {
    return velocityField;
  }

  /// Set a primary direction for the source distribution (tilted distribution).
  virtual std::optional<std::array<NumericType, 3>>
  getPrimaryDirection() const {
    return primaryDirection;
  }

  std::optional<std::string> getProcessName() const { return processName; }

  int getParticleLogSize(std::size_t particleIdx) const {
    return particleLogSize[particleIdx];
  }

  void setProcessName(std::string name) { processName = name; }

  virtual void
  setPrimaryDirection(const std::array<NumericType, 3> passedPrimaryDirection) {
    primaryDirection = rayInternal::Normalize(passedPrimaryDirection);
  }

  template <typename ParticleType>
  void insertNextParticleType(std::unique_ptr<ParticleType> &passedParticle,
                              const int dataLogSize = 0) {
    if (particles == nullptr) {
      particles = psSmartPointer<ParticleTypeList>::New();
    }
    particles->push_back(passedParticle->clone());
    particleLogSize.push_back(dataLogSize);
  }

  template <typename SurfaceModelType>
  void setSurfaceModel(psSmartPointer<SurfaceModelType> passedSurfaceModel) {
    surfaceModel = std::dynamic_pointer_cast<psSurfaceModel<NumericType>>(
        passedSurfaceModel);
  }

  template <typename AdvectionCallbackType>
  void setAdvectionCallback(
      psSmartPointer<AdvectionCallbackType> passedAdvectionCallback) {
    advectionCallback =
        std::dynamic_pointer_cast<psAdvectionCallback<NumericType, D>>(
            passedAdvectionCallback);
  }

  template <typename GeometricModelType>
  void
  setGeometricModel(psSmartPointer<GeometricModelType> passedGeometricModel) {
    geometricModel =
        std::dynamic_pointer_cast<psGeometricModel<NumericType, D>>(
            passedGeometricModel);
  }

  template <typename VelocityFieldType>
  void setVelocityField(psSmartPointer<VelocityFieldType> passedVelocityField) {
    velocityField = std::dynamic_pointer_cast<psVelocityField<NumericType>>(
        passedVelocityField);
  }
};
