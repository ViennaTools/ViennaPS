#pragma once

#include "psProcessModelBase.hpp"

#include <lsConcepts.hpp>

#include <rayParticle.hpp>
#include <raySource.hpp>

#include <vcCudaBuffer.hpp>

namespace viennaps {

using namespace viennacore;

/// The process model combines all models (particle types, surface model,
/// geometric model, advection callback)
template <typename NumericType, int D>
class ProcessModel : public ProcessModelBase<NumericType, D> {
protected:
  std::vector<std::unique_ptr<viennaray::AbstractParticle<NumericType>>>
      particles;
  SmartPointer<viennaray::Source<NumericType>> source = nullptr;
  std::vector<int> particleLogSize;
  std::optional<Vec3D<NumericType>> primaryDirection = std::nullopt;

public:
  auto &getParticleTypes() { return particles; }
  auto getSource() { return source; }
  bool useFluxEngine() override { return particles.size() > 0; }

  /// Set a primary direction for the source distribution (tilted distribution).
  virtual std::optional<Vec3D<NumericType>> getPrimaryDirection() const {
    return primaryDirection;
  }

  int getParticleLogSize(std::size_t particleIdx) const {
    return particleLogSize[particleIdx];
  }

  virtual void
  setPrimaryDirection(const Vec3D<NumericType> &passedPrimaryDirection) {
    primaryDirection = Normalize(passedPrimaryDirection);
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
};

} // namespace viennaps

#ifdef VIENNACORE_COMPILE_GPU
namespace viennaps::gpu {

using namespace viennacore;

template <class NumericType, int D>
class ProcessModel : public ProcessModelBase<NumericType, D> {
private:
  using ParticleTypeList = std::vector<viennaray::gpu::Particle<NumericType>>;

  ParticleTypeList particles;
  std::optional<std::string> processName = std::nullopt;
  std::optional<std::array<NumericType, 3>> primaryDirection = std::nullopt;
  std::string pipelineFileName;
  bool materialIds = false;

public:
  CudaBuffer processData;
  auto &getParticleTypes() { return particles; }
  auto getProcessDataDPtr() const { return processData.dPointer(); }
  bool useFluxEngine() override { return particles.size() > 0; }
  bool useMaterialIds() const { return materialIds; }
  void setUseMaterialIds(bool passedMaterialIds) {
    materialIds = passedMaterialIds;
  }

  void setPipelineFileName(const std::string &fileName) {
    pipelineFileName = fileName;
  }
  auto getPipelineFileName() const { return pipelineFileName; }

  void insertNextParticleType(
      const viennaray::gpu::Particle<NumericType> &passedParticle) {
    particles.push_back(passedParticle);
  }

  /// Set a primary direction for the source distribution (tilted distribution).
  virtual std::optional<std::array<NumericType, 3>>
  getPrimaryDirection() const {
    return primaryDirection;
  }

  virtual void
  setPrimaryDirection(const std::array<NumericType, 3> passedPrimaryDirection) {
    primaryDirection = Normalize(passedPrimaryDirection);
  }
};

} // namespace viennaps::gpu
#endif