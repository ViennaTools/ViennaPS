#pragma once

#include <context.hpp>

#include <curtParticle.hpp>

#include <psAdvectionCallback.hpp>
#include <psGeometricModel.hpp>
#include <psSurfaceModel.hpp>
#include <psVelocityField.hpp>

namespace viennaps {

namespace gpu {

using namespace viennacore;

template <typename NumericType> class ProcessModel {
private:
  using ParticleTypeList = std::vector<Particle<NumericType>>;

  ParticleTypeList particles;
  SmartPointer<::viennaps::SurfaceModel<NumericType>> surfaceModel = nullptr;
  SmartPointer<VelocityField<NumericType>> velocityField = nullptr;
  std::string processName = "default";
  std::string pipelineFileName = "";

public:
  auto getParticleTypes() { return particles; }
  auto getSurfaceModel() { return surfaceModel; }
  auto getVelocityField() { return velocityField; }

  void setProcessName(std::string name) { processName = name; }
  auto getProcessName() const { return processName; }

  void setPipelineFileName(std::string fileName) {
    pipelineFileName = fileName;
  }
  auto getPipelineFileName() const { return pipelineFileName; }

  void insertNextParticleType(const Particle<NumericType> &passedParticle) {
    particles.push_back(passedParticle);
  }

  void setSurfaceModel(
      SmartPointer<::viennaps::SurfaceModel<NumericType>> passedSurfaceModel) {
    surfaceModel = passedSurfaceModel;
  }

  void setVelocityField(
      SmartPointer<VelocityField<NumericType>> passedVelocityField) {
    velocityField = passedVelocityField;
  }
};

} // namespace gpu

} // namespace viennaps