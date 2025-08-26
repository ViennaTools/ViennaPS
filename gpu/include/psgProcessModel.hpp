#pragma once

#include <vcCudaBuffer.hpp>

#include <raygParticle.hpp>

#include <process/psProcessModelBase.hpp>

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
