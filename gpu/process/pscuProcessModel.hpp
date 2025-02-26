#pragma once

#include <gpu/vcContext.hpp>
#include <gpu/vcCudaBuffer.hpp>

#include <curtParticle.hpp>

#include <psProcessModelBase.hpp>

namespace viennaps {

namespace gpu {

using namespace viennacore;

template <class NumericType, int D>
class ProcessModel : public ProcessModelBase<NumericType, D> {
private:
  using ParticleTypeList = std::vector<Particle<NumericType>>;

  ParticleTypeList particles;
  std::optional<std::string> processName = std::nullopt;
  std::optional<std::array<NumericType, 3>> primaryDirection = std::nullopt;
  std::string pipelineFileName;

public:
  CudaBuffer processData;
  auto &getParticleTypes() { return particles; }
  auto getProcessDataDPtr() { return processData.dPointer(); }
  bool useFluxEngine() override { return particles.size() > 0; }

  void setPipelineFileName(const std::string &fileName) {
    pipelineFileName = fileName;
  }
  auto getPipelineFileName() const { return pipelineFileName; }

  void insertNextParticleType(const Particle<NumericType> &passedParticle) {
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

} // namespace gpu
} // namespace viennaps