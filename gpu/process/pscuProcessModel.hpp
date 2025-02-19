#pragma once

#include <gpu/vcContext.hpp>
#include <gpu/vcCudaBuffer.hpp>

#include <curtParticle.hpp>

#include <psAdvectionCallback.hpp>
#include <psGeometricModel.hpp>
#include <psSurfaceModel.hpp>
#include <psVelocityField.hpp>

namespace viennaps {

namespace gpu {

using namespace viennacore;

template <class NumericType, int D> class ProcessModel {
private:
  using ParticleTypeList = std::vector<Particle<NumericType>>;

  ParticleTypeList particles;
  SmartPointer<::viennaps::SurfaceModel<NumericType>> surfaceModel = nullptr;
  SmartPointer<VelocityField<NumericType, D>> velocityField = nullptr;
  std::optional<std::string> processName = std::nullopt;
  std::optional<std::array<NumericType, 3>> primaryDirection = std::nullopt;
  std::string pipelineFileName;

public:
  CudaBuffer processData;
  auto &getParticleTypes() { return particles; }
  auto &getSurfaceModel() { return surfaceModel; }
  auto &getVelocityField() { return velocityField; }
  auto getProcessDataDPtr() { return processData.dPointer(); }

  void setProcessName(const std::string &name) { processName = name; }
  auto getProcessName() const { return processName; }

  void setPipelineFileName(const std::string &fileName) {
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
      SmartPointer<VelocityField<NumericType, D>> passedVelocityField) {
    velocityField = passedVelocityField;
  }

  virtual void initialize(SmartPointer<Domain<NumericType, D>> domain,
                          const NumericType processDuration) {}
  virtual void reset() {}
};

} // namespace gpu
} // namespace viennaps