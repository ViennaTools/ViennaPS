#pragma once

#include <context.hpp>

#include <curtParticle.hpp>

#include <psAdvectionCallback.hpp>
#include <psGeometricModel.hpp>
#include <psSmartPointer.hpp>
#include <psSurfaceModel.hpp>
#include <psVelocityField.hpp>

template <typename NumericType> class pscuProcessModel {
private:
  using ParticleTypeList = std::vector<curtParticle<NumericType>>;

  psSmartPointer<ParticleTypeList> particles = nullptr;
  psSmartPointer<psSurfaceModel<NumericType>> surfaceModel = nullptr;
  psSmartPointer<psAdvectionCallback<NumericType, DIM>> advectionCallback =
      nullptr;
  psSmartPointer<psGeometricModel<NumericType, DIM>> geometricModel = nullptr;
  psSmartPointer<psVelocityField<NumericType>> velocityField = nullptr;
  std::string processName = "default";
  char *embbededPtxCode = nullptr;

public:
  virtual psSmartPointer<ParticleTypeList> getParticleTypes() {
    return particles;
  }
  virtual psSmartPointer<psSurfaceModel<NumericType>> getSurfaceModel() {
    return surfaceModel;
  }
  virtual psSmartPointer<psAdvectionCallback<NumericType, DIM>>
  getAdvectionCallback() {
    return advectionCallback;
  }
  virtual psSmartPointer<psGeometricModel<NumericType, DIM>>
  getGeometricModel() {
    return geometricModel;
  }
  virtual psSmartPointer<psVelocityField<NumericType>> getVelocityField() {
    return velocityField;
  }

  void setProcessName(std::string name) { processName = name; }

  std::string getProcessName() { return processName; }

  void setPtxCode(char *ptxCode) { embbededPtxCode = ptxCode; }
  char *getPtxCode() { return embbededPtxCode; }

  void insertNextParticleType(const curtParticle<NumericType> &passedParticle) {
    if (particles == nullptr) {
      particles = psSmartPointer<ParticleTypeList>::New();
    }

    particles->push_back(passedParticle);
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
        std::dynamic_pointer_cast<psAdvectionCallback<NumericType, DIM>>(
            passedAdvectionCallback);
  }

  template <typename GeometricModelType>
  void
  setGeometricModel(psSmartPointer<GeometricModelType> passedGeometricModel) {
    geometricModel =
        std::dynamic_pointer_cast<psGeometricModel<NumericType, DIM>>(
            passedGeometricModel);
  }

  template <typename VelocityFieldType>
  void setVelocityField(psSmartPointer<VelocityFieldType> passedVelocityField) {
    velocityField = std::dynamic_pointer_cast<psVelocityField<NumericType>>(
        passedVelocityField);
  }
};
