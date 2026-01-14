#pragma once

#include "../psDomain.hpp"
#include "psAdvectionCallback.hpp"
#include "psGeometricModel.hpp"
#include "psSurfaceModel.hpp"
#include "psVelocityField.hpp"

#include <lsConcepts.hpp>

#include <rayParticle.hpp>
#include <raySource.hpp>

#ifdef VIENNACORE_COMPILE_GPU
#include <gpu/raygCallableConfig.hpp>
#include <vcCudaBuffer.hpp>
#endif

#include <vcLogger.hpp>

namespace viennaps {

using namespace viennacore;

/// The process model combines all models (particle types, surface model,
/// geometric model, advection callback)
template <typename NumericType, int D> class ProcessModelBase {
protected:
  SmartPointer<SurfaceModel<NumericType>> surfaceModel = nullptr;
  SmartPointer<AdvectionCallback<NumericType, D>> advectionCallback = nullptr;
  SmartPointer<GeometricModel<NumericType, D>> geometricModel = nullptr;
  SmartPointer<VelocityField<NumericType, D>> velocityField = nullptr;
  std::optional<std::string> processName = std::nullopt;
  std::unordered_map<std::string, std::vector<double>> processMetaData;

  bool hasGPU = false; // indicates whether a GPU version of the model exists
  bool isALP = false;  // indicates whether the model is an atomic layer process

public:
  virtual ~ProcessModelBase() = default;

  virtual void initialize(SmartPointer<Domain<NumericType, D>> domain,
                          const NumericType processDuration) {}
  virtual void finalize(SmartPointer<Domain<NumericType, D>> domain,
                        const NumericType processedDuration) {}
  virtual bool useFluxEngine() { return false; }
  virtual SmartPointer<ProcessModelBase<NumericType, D>> getGPUModel() {
    VIENNACORE_LOG_WARNING("Cannot convert CPU model to GPU model.");
    return nullptr;
  }

  auto getSurfaceModel() const { return surfaceModel; }
  auto getAdvectionCallback() const { return advectionCallback; }
  auto getGeometricModel() const { return geometricModel; }
  auto getVelocityField() const { return velocityField; }
  auto getProcessName() const { return processName; }
  auto getProcessMetaData() const { return processMetaData; }
  auto isALPModel() const { return isALP; }
  auto hasGPUModel() const { return hasGPU; }

  void setProcessName(const std::string &name) { processName = name; }

  void
  setSurfaceModel(SmartPointer<SurfaceModel<NumericType>> passedSurfaceModel) {
    surfaceModel = passedSurfaceModel;
  }

  void setAdvectionCallback(
      SmartPointer<AdvectionCallback<NumericType, D>> passedAdvectionCallback) {
    advectionCallback = passedAdvectionCallback;
  }

  void setGeometricModel(
      SmartPointer<GeometricModel<NumericType, D>> passedGeometricModel) {
    geometricModel = passedGeometricModel;
  }

  void setVelocityField(
      SmartPointer<VelocityField<NumericType, D>> passedVelocityField) {
    velocityField = passedVelocityField;
  }
};

/// Process model for CPU-based particle tracing (or no particle tracing)
template <typename NumericType, int D>
class ProcessModelCPU : public ProcessModelBase<NumericType, D> {
protected:
  std::vector<std::unique_ptr<viennaray::AbstractParticle<NumericType>>>
      particles;
  SmartPointer<viennaray::Source<NumericType>> source = nullptr;
  std::vector<int> particleLogSize;
  std::optional<Vec3D<NumericType>> primaryDirection = std::nullopt;
  std::vector<viennaray::DataLog<NumericType>> particleDataLogs_;

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

  void initializeParticleDataLogs() {
    particleDataLogs_.resize(particles.size());
    for (std::size_t i = 0; i < particles.size(); i++) {
      if (int logSize = particleLogSize[i]; logSize > 0) {
        particleDataLogs_[i].data.resize(1);
        particleDataLogs_[i].data[0].resize(logSize);
      }
    }
  }

  void mergeParticleData(viennaray::DataLog<NumericType> &log,
                         size_t particleIdx) {
    if (particleIdx < particleDataLogs_.size()) {
      particleDataLogs_[particleIdx].merge(log);
    }
  }

  void writeParticleDataLogs(const std::string &fileName) {
    std::ofstream file(fileName.c_str());

    for (std::size_t i = 0; i < particleDataLogs_.size(); i++) {
      if (!particleDataLogs_[i].data.empty()) {
        file << "particle" << i << "_data\n";
        for (std::size_t j = 0; j < particleDataLogs_[i].data[0].size(); j++) {
          file << particleDataLogs_[i].data[0][j] << " ";
        }
        file << "\n";
      }
    }

    file.close();
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

  auto getParticleDataLabels() const {
    std::vector<std::string> dataLabels;
    for (size_t pIdx = 0; pIdx < particles.size(); pIdx++) {
      auto labels = particles[pIdx]->getLocalDataLabels();
      for (size_t dIdx = 0; dIdx < labels.size(); dIdx++) {
        dataLabels.push_back(labels[dIdx]);
      }
    }
    return dataLabels;
  }
};

} // namespace viennaps

#ifdef VIENNACORE_COMPILE_GPU
namespace viennaps::gpu {

using namespace viennacore;

template <class NumericType, int D>
class ProcessModelGPU : public ProcessModelBase<NumericType, D> {
private:
  std::vector<viennaray::gpu::Particle<NumericType>> particles;
  std::optional<std::array<NumericType, 3>> primaryDirection = std::nullopt;
  std::string callableFileName = "ViennaPSCallableWrapper";
  bool materialIds = false;
  std::unordered_map<std::string, unsigned> particleMap_;
  std::vector<viennaray::gpu::CallableConfig> callableMap_;

public:
  CudaBuffer processData;
  auto &getParticleTypes() { return particles; }
  auto getProcessDataDPtr() const { return processData.dPointer(); }
  bool useMaterialIds() const { return materialIds; }
  void setUseMaterialIds(bool passedMaterialIds) {
    materialIds = passedMaterialIds;
  }
  bool useFluxEngine() override { return particles.size() > 0; }

  void setCallableFileName(const std::string &fileName) {
    callableFileName = fileName;
  }
  auto getCallableFileName() const { return callableFileName; }

  void insertNextParticleType(
      const viennaray::gpu::Particle<NumericType> &passedParticle) {
    particles.push_back(passedParticle);
  }

  void setParticleCallableMap(
      const std::unordered_map<std::string, unsigned> &pMap,
      const std::vector<viennaray::gpu::CallableConfig> &cMap) {
    particleMap_ = pMap;
    callableMap_ = cMap;
  }

  std::tuple<std::unordered_map<std::string, unsigned>,
             std::vector<viennaray::gpu::CallableConfig>>
  getParticleCallableMap() const {
    return {particleMap_, callableMap_};
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

  auto getParticleDataLabels() const {
    std::vector<std::string> dataLabels;
    for (size_t pIdx = 0; pIdx < particles.size(); pIdx++) {
      for (size_t dIdx = 0; dIdx < particles[pIdx].dataLabels.size(); dIdx++) {
        dataLabels.push_back(particles[pIdx].dataLabels[dIdx]);
      }
    }
    return dataLabels;
  }
};

} // namespace viennaps::gpu
#endif