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
  using ParticleTypeList = std::vector<curtParticle<NumericType>>;

  SmartPointer<ParticleTypeList> particles = nullptr;
  SmartPointer<SurfaceModel<NumericType>> surfaceModel = nullptr;
  SmartPointer<AdvectionCallback<NumericType, DIM>> advectionCallback = nullptr;
  SmartPointer<GeometricModel<NumericType, DIM>> geometricModel = nullptr;
  SmartPointer<VelocityField<NumericType>> velocityField = nullptr;
  std::string processName = "default";
  char *embbededPtxCode = nullptr;

public:
  virtual SmartPointer<ParticleTypeList> getParticleTypes() {
    return particles;
  }
  virtual SmartPointer<SurfaceModel<NumericType>> getSurfaceModel() {
    return surfaceModel;
  }
  virtual SmartPointer<AdvectionCallback<NumericType, DIM>>
  getAdvectionCallback() {
    return advectionCallback;
  }
  virtual SmartPointer<GeometricModel<NumericType, DIM>> getGeometricModel() {
    return geometricModel;
  }
  virtual SmartPointer<VelocityField<NumericType>> getVelocityField() {
    return velocityField;
  }

  void setProcessName(std::string name) { processName = name; }

  std::string getProcessName() { return processName; }

  void setPtxCode(char *ptxCode) { embbededPtxCode = ptxCode; }
  char *getPtxCode() { return embbededPtxCode; }

  void insertNextParticleType(const curtParticle<NumericType> &passedParticle) {
    if (particles == nullptr) {
      particles = SmartPointer<ParticleTypeList>::New();
    }

    particles->push_back(passedParticle);
  }

  template <typename SurfaceModelType>
  void setSurfaceModel(SmartPointer<SurfaceModelType> passedSurfaceModel) {
    surfaceModel = std::dynamic_pointer_cast<SurfaceModel<NumericType>>(
        passedSurfaceModel);
  }

  template <typename AdvectionCallbackType>
  void setAdvectionCallback(
      SmartPointer<AdvectionCallbackType> passedAdvectionCallback) {
    advectionCallback =
        std::dynamic_pointer_cast<AdvectionCallback<NumericType, DIM>>(
            passedAdvectionCallback);
  }

  template <typename GeometricModelType>
  void
  setGeometricModel(SmartPointer<GeometricModelType> passedGeometricModel) {
    geometricModel =
        std::dynamic_pointer_cast<GeometricModel<NumericType, DIM>>(
            passedGeometricModel);
  }

  template <typename VelocityFieldType>
  void setVelocityField(SmartPointer<VelocityFieldType> passedVelocityField) {
    velocityField = std::dynamic_pointer_cast<VelocityField<NumericType>>(
        passedVelocityField);
  }
};

} // namespace gpu

} // namespace viennaps