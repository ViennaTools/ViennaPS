#pragma once

#include "psAdvectionCallback.hpp"
#include "psDomain.hpp"
#include "psGeometricModel.hpp"
#include "psSurfaceModel.hpp"
#include "psVelocityField.hpp"

#include <lsConcepts.hpp>
#include <lsPointData.hpp>

#include <rayParticle.hpp>
#include <raySource.hpp>

#include <vcSmartPointer.hpp>

namespace viennaps {

using namespace viennacore;

/// The process model combines all models (particle types, surface model,
/// geometric model, advection callback)
template <typename NumericType, int D> class ProcessModel {
protected:
  std::vector<std::unique_ptr<viennaray::AbstractParticle<NumericType>>>
      particles;
  SmartPointer<viennaray::Source<NumericType>> source = nullptr;
  std::vector<int> particleLogSize;
  SmartPointer<SurfaceModel<NumericType>> surfaceModel = nullptr;
  SmartPointer<AdvectionCallback<NumericType, D>> advectionCallback = nullptr;
  SmartPointer<GeometricModel<NumericType, D>> geometricModel = nullptr;
  SmartPointer<VelocityField<NumericType, D>> velocityField = nullptr;
  std::optional<std::string> processName = std::nullopt;
  std::optional<std::array<NumericType, 3>> primaryDirection = std::nullopt;

public:
  virtual ~ProcessModel() = default;

  virtual void initialize(SmartPointer<Domain<NumericType, D>> domain,
                          const NumericType processDuration) {}
  virtual void reset() {}

  auto &getParticleTypes() { return particles; }
  auto getSurfaceModel() const { return surfaceModel; }
  auto getAdvectionCallback() const { return advectionCallback; }
  auto getGeometricModel() const { return geometricModel; }
  auto getVelocityField() const { return velocityField; }
  auto getSource() { return source; }
  auto getProcessName() const { return processName; }

  /// Set a primary direction for the source distribution (tilted distribution).
  virtual std::optional<std::array<NumericType, 3>>
  getPrimaryDirection() const {
    return primaryDirection;
  }

  int getParticleLogSize(std::size_t particleIdx) const {
    return particleLogSize[particleIdx];
  }

  void setProcessName(const std::string &name) { processName = name; }

  virtual void
  setPrimaryDirection(const std::array<NumericType, 3> passedPrimaryDirection) {
    primaryDirection = rayInternal::Normalize(passedPrimaryDirection);
  }

  template <typename ParticleType,
            lsConcepts::IsBaseOf<viennaray::Particle<ParticleType, NumericType>,
                                 ParticleType> = lsConcepts::assignable>
  void insertNextParticleType(std::unique_ptr<ParticleType> &passedParticle,
                              const int dataLogSize = 0) {
    particles.push_back(passedParticle->clone());
    particleLogSize.push_back(dataLogSize);
  }

  void setSource(SmartPointer<viennaray::Source<NumericType>> passedSource) {
    source = passedSource;
  }

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
