#pragma once

#include "psAdvectionCallback.hpp"
#include "psGeometricModel.hpp"
#include "psSmartPointer.hpp"
#include "psSurfaceModel.hpp"
#include "psVelocityField.hpp"

#include <lsConcepts.hpp>
#include <rayParticle.hpp>
#include <raySource.hpp>

/// The process model combines all models (particle types, surface model,
/// geometric model, advection callback)
template <typename NumericType, int D> class psProcessModel {
protected:
  using ParticleTypeList =
      std::vector<std::unique_ptr<rayAbstractParticle<NumericType>>>;

  ParticleTypeList particles;
  psSmartPointer<raySource<NumericType>> source = nullptr;
  std::vector<int> particleLogSize;
  psSmartPointer<psSurfaceModel<NumericType>> surfaceModel = nullptr;
  psSmartPointer<psAdvectionCallback<NumericType, D>> advectionCallback =
      nullptr;
  psSmartPointer<psGeometricModel<NumericType, D>> geometricModel = nullptr;
  psSmartPointer<psVelocityField<NumericType>> velocityField = nullptr;
  std::optional<std::string> processName = std::nullopt;
  std::optional<std::array<NumericType, 3>> primaryDirection = std::nullopt;

public:
  virtual ~psProcessModel() = default;

  ParticleTypeList const &getParticleTypes() const { return particles; }
  psSmartPointer<psSurfaceModel<NumericType>> getSurfaceModel() const {
    return surfaceModel;
  }
  psSmartPointer<psAdvectionCallback<NumericType, D>>
  getAdvectionCallback() const {
    return advectionCallback;
  }
  psSmartPointer<psGeometricModel<NumericType, D>> getGeometricModel() const {
    return geometricModel;
  }
  psSmartPointer<psVelocityField<NumericType>> getVelocityField() const {
    return velocityField;
  }
  psSmartPointer<raySource<NumericType>> getSource() { return source; }

  /// Set a primary direction for the source distribution (tilted distribution).
  virtual std::optional<std::array<NumericType, 3>>
  getPrimaryDirection() const {
    return primaryDirection;
  }

  std::optional<std::string> getProcessName() const { return processName; }

  int getParticleLogSize(std::size_t particleIdx) const {
    return particleLogSize[particleIdx];
  }

  void setProcessName(std::string name) { processName = std::move(name); }

  virtual void
  setPrimaryDirection(const std::array<NumericType, 3> passedPrimaryDirection) {
    primaryDirection = rayInternal::Normalize(passedPrimaryDirection);
  }

  template <typename ParticleType,
            lsConcepts::IsBaseOf<rayParticle<ParticleType, NumericType>,
                                 ParticleType> = lsConcepts::assignable>
  void insertNextParticleType(std::unique_ptr<ParticleType> &passedParticle,
                              const int dataLogSize = 0) {
    particles.push_back(passedParticle->clone());
    particleLogSize.push_back(dataLogSize);
  }

  void setSource(psSmartPointer<raySource<NumericType>> passedSource) {
    source = passedSource;
  }

  void setSurfaceModel(
      psSmartPointer<psSurfaceModel<NumericType>> passedSurfaceModel) {
    surfaceModel = std::dynamic_pointer_cast<psSurfaceModel<NumericType>>(
        passedSurfaceModel);
  }

  void setAdvectionCallback(psSmartPointer<psAdvectionCallback<NumericType, D>>
                                passedAdvectionCallback) {
    advectionCallback = passedAdvectionCallback;
  }

  void setGeometricModel(
      psSmartPointer<psGeometricModel<NumericType, D>> passedGeometricModel) {
    geometricModel = passedGeometricModel;
  }

  void setVelocityField(
      psSmartPointer<psVelocityField<NumericType>> passedVelocityField) {
    velocityField = passedVelocityField;
  }
};
