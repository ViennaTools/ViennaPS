#ifndef PS_PROCESS_MODEL
#define PS_PROCESS_MODEL

#include <psSmartPointer.hpp>
#include <psSurfaceModel.hpp>
#include <psVelocityField.hpp>
#include <psVolumeModel.hpp>
#include <rayParticle.hpp>

template <typename NumericType> class psProcessModel {
private:
  std::vector<std::unique_ptr<rayAbstractParticle<NumericType>>> particles;
  psSmartPointer<psSurfaceModel<NumericType>> surfaceModel = nullptr;
  psSmartPointer<psVolumeModel<NumericType>> volumeModel = nullptr;

  // return empty vector in base implementation
  // use smart pointers
  virtual std::vector<std::unique_ptr<rayAbstractParticle<NumericType>>>
  getParticleTypes() {
    return std::vector<std::unique_ptr<rayAbstractParticle<NumericType>>>();
  }
  virtual std::unique_ptr<psSurfaceModel<NumericType>> getSurfaceModel() {
    return std::make_unique<psSurfaceModel<NumericType>>();
  }
  virtual psSmartPointer<psVolumeModel<NumericType>> getVolumeModel() {
    return psSmartPointer<psVolumeModel<NumericType>>::New();
  }
  virtual psSmartPointer<psVelocityField<NumericType>> getVelocityField() {
    return psSmartPointer<psVelocityField<NumericType>>::New();
  }
  template <typename ParticleType>
  void setNextParticleType(std::unique_ptr<ParticleType> &passedParticle) {
    auto particle = passedParticle->clone();
    particles.push_back(particle);
  }
  void setSurfaceModel(
      psSmartPointer<psSurfaceModel<NumericType>> &passedSurfaceModel) {
    surfaceModel = passedSurfaceModel;
  }
  void setVolumeModel(
      psSmartPointer<psVolumeModel<NumericType>> &passedVolumeModel) {
    volumeModel = passedVolumeModel;
  }
};

#endif