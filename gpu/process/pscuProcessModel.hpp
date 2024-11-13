#pragma once

#include <context.hpp>

#include <curtParticle.hpp>

#include <psAdvectionCallback.hpp>
#include <psGeometricModel.hpp>
#include <psSurfaceModel.hpp>
#include <psVelocityField.hpp>

// #include <pscuSurfaceModel.hpp>

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
  char *embbededPtxCode = nullptr;

public:
  auto getParticleTypes() { return particles; }
  auto getSurfaceModel() { return surfaceModel; }
  auto getVelocityField() { return velocityField; }

  void setProcessName(std::string name) { processName = name; }

  std::string getProcessName() { return processName; }

  void setPtxCode(char *ptxCode) { embbededPtxCode = ptxCode; }
  char *getPtxCode() { return embbededPtxCode; }

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