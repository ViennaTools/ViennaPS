#pragma once

#include <curtParticle.hpp>

#include <pscuSurfaceModel.hpp>

#include <psSmartPointer.hpp>
#include <psVelocityField.hpp>
#include <psVolumeModel.hpp>

template <typename NumericType> class pscuProcessModel {
private:
  using ParticleTypeList = std::vector<curtParticle<NumericType>>;

  psSmartPointer<ParticleTypeList> particles = nullptr;
  psSmartPointer<pscuSurfaceModel<NumericType>> surfaceModel = nullptr;
  psSmartPointer<psVolumeModel<NumericType, 3>> volumeModel = nullptr;
  psSmartPointer<psVelocityField<NumericType>> velocityField = nullptr;
  std::string processName = "default";
  char *embbededPtxCode = nullptr;

public:
  virtual psSmartPointer<ParticleTypeList> getParticleTypes() {
    return particles;
  }
  virtual psSmartPointer<pscuSurfaceModel<NumericType>> getSurfaceModel() {
    return surfaceModel;
  }
  virtual psSmartPointer<psVolumeModel<NumericType, 3>> getVolumeModel() {
    return volumeModel;
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
    surfaceModel = std::dynamic_pointer_cast<pscuSurfaceModel<NumericType>>(
        passedSurfaceModel);
  }

  template <typename VolumeModelType>
  void setVolumeModel(psSmartPointer<VolumeModelType> passedVolumeModel) {
    volumeModel = std::dynamic_pointer_cast<psVolumeModel<NumericType, 3>>(
        passedVolumeModel);
  }

  template <typename VelocityFieldType>
  void setVelocityField(psSmartPointer<VelocityFieldType> passedVelocityField) {
    velocityField = std::dynamic_pointer_cast<psVelocityField<NumericType>>(
        passedVelocityField);
  }
};
