#ifndef PS_PROCESS_MODEL
#define PS_PROCESS_MODEL

#include <psSmartPointer.hpp>
#include <psSurfaceModel.hpp>
#include <psVelocityField.hpp>
#include <psVolumeModel.hpp>
#include <rayParticle.hpp>

template <typename NumericType> class psProcessModel {
private:
  using ParticleTypeList =
      std::vector<std::unique_ptr<rayAbstractParticle<NumericType>>>;

  psSmartPointer<ParticleTypeList> particles = nullptr;
  psSmartPointer<psSurfaceModel<NumericType>> surfaceModel = nullptr;
  psSmartPointer<psVolumeModel<NumericType>> volumeModel = nullptr;
  psSmartPointer<psVelocityField<NumericType>> velocityField = nullptr;
  std::string processName = "default";

public:
  virtual psSmartPointer<ParticleTypeList> getParticleTypes() {
    return particles;
  }
  virtual psSmartPointer<psSurfaceModel<NumericType>> getSurfaceModel() {
    return surfaceModel;
  }
  virtual psSmartPointer<psVolumeModel<NumericType>> getVolumeModel() {
    return volumeModel;
  }
  virtual psSmartPointer<psVelocityField<NumericType>> getVelocityField() {
    return velocityField;
  }

  void setProcessName(std::string name) { processName = name; }

  std::string getProcessName() { return processName; }

  template <typename ParticleType>
  void insertNextParticleType(std::unique_ptr<ParticleType> &passedParticle) {
    if (particles == nullptr) {
      particles = psSmartPointer<ParticleTypeList>::New();
    }
    particles->push_back(passedParticle->clone());
  }

  template <typename SurfaceModelType>
  void setSurfaceModel(psSmartPointer<SurfaceModelType> passedSurfaceModel) {
    surfaceModel = std::dynamic_pointer_cast<psSurfaceModel<NumericType>>(
        passedSurfaceModel);
  }

  template <typename VolumeModelType>
  void setVolumeModel(psSmartPointer<VolumeModelType> passedVolumeModel) {
    volumeModel = std::dynamic_pointer_cast<psVolumeModel<NumericType>>(
        passedVolumeModel);
  }

  template <typename VelocityFieldType>
  void setVelocityField(psSmartPointer<VelocityFieldType> passedVelocityField) {
    velocityField = std::dynamic_pointer_cast<psVelocityField<NumericType>>(
        passedVelocityField);
  }
};

#endif