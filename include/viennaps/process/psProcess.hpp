#pragma once

#include "../psPreCompileMacros.hpp"

#include "psProcessContext.hpp"

// Process strategies
#include "psALPStrategy.hpp"
#include "psAnalyticProcessStrategy.hpp"
#include "psCallbackOnlyStrategy.hpp"
#include "psFluxProcessStrategy.hpp"
#include "psGeometricProcessStrategy.hpp"

// Flux engines
#include "psCPUDiskEngine.hpp"
#include "psCPUTriangleEngine.hpp"
#include "psGPUDiskEngine.hpp"
#include "psGPULineEngine.hpp"
#include "psGPUTriangleEngine.hpp"

namespace viennaps {

using namespace viennacore;

#ifdef VIENNACORE_COMPILE_GPU
inline constexpr bool gpuAvailable() { return true; }
#else
inline constexpr bool gpuAvailable() { return false; }
#endif

template <typename T> constexpr bool always_false = false;

template <typename NumericType, int D> class Process {
private:
  ProcessContext<NumericType, D> context_;
  std::vector<std::unique_ptr<ProcessStrategy<NumericType, D>>> strategies_;
  FluxEngineType fluxEngineType_ = FluxEngineType::AUTO;

public:
  Process() { initializeStrategies(); }
  Process(SmartPointer<Domain<NumericType, D>> domain) : context_{domain} {
    initializeStrategies();
  }
  template <typename... ParamArgs>
  Process(SmartPointer<Domain<NumericType, D>> domain,
          SmartPointer<ProcessModelBase<NumericType, D>> model,
          NumericType processDuration = 0., ParamArgs... params)
      : context_{domain, model, processDuration} {
    (setParameters(params), ...);
    initializeStrategies();
  }

  void setDomain(SmartPointer<Domain<NumericType, D>> domain) {
    context_.domain = domain;
  }

  void setProcessModel(SmartPointer<ProcessModelBase<NumericType, D>> model) {
    context_.model = model;
  }

  void setProcessDuration(double duration) {
    context_.processDuration = duration;
  }

  template <typename ParamType> void setParameters(const ParamType &params) {
    if constexpr (std::is_same_v<ParamType, RayTracingParameters>) {
      context_.rayTracingParams = params;
    } else if constexpr (std::is_same_v<ParamType, AdvectionParameters>) {
      context_.advectionParams = params;
    } else if constexpr (std::is_same_v<ParamType, CoverageParameters>) {
      context_.coverageParams = params;
    } else if constexpr (std::is_same_v<ParamType,
                                        AtomicLayerProcessParameters>) {
      context_.atomicLayerParams = params;
    } else {
      static_assert(always_false<ParamType>,
                    "Unsupported parameter type for Process.");
    }
  }

  void setFluxEngineType(FluxEngineType type) { fluxEngineType_ = type; }

  void setIntermediateOutputPath(const std::string &path) {
    context_.intermediateOutputPath = path;
  }

  void apply() {

    if (!checkInputUpdateContext())
      return;

    // Find appropriate strategy
    auto strategy = findStrategy();
    if (!strategy) {
      VIENNACORE_LOG_ERROR(
          "No suitable strategy found for process configuration.");
      return;
    }
    VIENNACORE_LOG_DEBUG("Using strategy: " + std::string(strategy->name()));

    if (strategy->requiresFluxEngine()) {
      VIENNACORE_LOG_DEBUG("Setting up " + to_string(fluxEngineType_) +
                           " flux engine for strategy.");
      strategy->setFluxEngine(createFluxEngine());
    }

    // Execute strategy
    context_.resetTime(); // Reset process time and previous time step
    auto result = strategy->execute(context_);
    handleProcessResult(result);

    if (static_cast<int>(context_.domain->getMetaDataLevel()) >=
        static_cast<int>(MetaDataLevel::PROCESS)) {
      context_.domain->addMetaData(context_.model->getProcessMetaData());
    }
  }

  SmartPointer<viennals::Mesh<NumericType>> calculateFlux() {
    if (!checkInputUpdateContext())
      return nullptr;

    const auto name = context_.getProcessName();
    if (!context_.flags.useFluxEngine) {
      VIENNACORE_LOG_ERROR("Process model '" + name +
                           "' does not use flux engine.");
      return nullptr;
    }

    auto strategy = std::make_unique<FluxProcessStrategy<NumericType, D>>(
        createFluxEngine());
    strategy->calculateFlux(context_);

    return context_.diskMesh;
  }

private:
  void initializeStrategies() {
    strategies_.clear();
    // Add strategies in priority order
    strategies_.push_back(
        std::make_unique<GeometricProcessStrategy<NumericType, D>>());
    strategies_.push_back(
        std::make_unique<CallbackOnlyStrategy<NumericType, D>>());
    strategies_.push_back(
        std::make_unique<AnalyticProcessStrategy<NumericType, D>>());
    strategies_.push_back(
        std::make_unique<FluxProcessStrategy<NumericType, D>>());
    strategies_.push_back(std::make_unique<ALPStrategy<NumericType, D>>());
  }

  ProcessStrategy<NumericType, D> *findStrategy() const {
    for (auto &strategy : strategies_) {
      if (strategy->canHandle(context_)) {
        return strategy.get();
      }
    }
    return nullptr;
  }

  static void handleProcessResult(const ProcessResult result) {
    switch (result) {
    case ProcessResult::SUCCESS:
      // Handle success
      break;
    case ProcessResult::EARLY_TERMINATION:
      VIENNACORE_LOG_INFO("Process terminated early.");
      break;
    case ProcessResult::USER_INTERRUPTED:
      VIENNACORE_LOG_INFO("Process interrupted by user.");
#ifdef VIENNATOOLS_PYTHON_BUILD
      throw pybind11::error_already_set();
#endif
      break;
    case ProcessResult::INVALID_INPUT:
      VIENNACORE_LOG_ERROR("Invalid input for process.");
      break;
    case ProcessResult::FAILURE:
      VIENNACORE_LOG_ERROR("Process failed.");
      break;
    case ProcessResult::NOT_IMPLEMENTED:
      VIENNACORE_LOG_ERROR("Process feature not implemented.");
      break;
    case ProcessResult::CONVERGENCE_FAILURE:
      VIENNACORE_LOG_ERROR("Process failed to converge.");
      break;
    }
  }

  // Factory method for creating flux engines
  std::unique_ptr<FluxEngine<NumericType, D>> createFluxEngine() const {
    assert(fluxEngineType_ != FluxEngineType::AUTO &&
           "Flux engine type must be specified before creation.");
    VIENNACORE_LOG_DEBUG("Creating flux engine of type: " +
                         to_string(fluxEngineType_));
    // Create CPU engine
    if (fluxEngineType_ == FluxEngineType::CPU_DISK) {
      return std::make_unique<CPUDiskEngine<NumericType, D>>();
    } else if (fluxEngineType_ == FluxEngineType::CPU_TRIANGLE) {
      return std::make_unique<CPUTriangleEngine<NumericType, D>>();
    }

    // Create GPU engine
    return createGPUFluxEngine();
  }

private:
  std::unique_ptr<FluxEngine<NumericType, D>> createGPUFluxEngine() const {
#ifndef VIENNACORE_COMPILE_GPU
    VIENNACORE_LOG_ERROR("GPU support not compiled in ViennaPS.");
    return nullptr;
#else
    auto deviceContext = getOrCreateDeviceContext();
    if (!deviceContext) {
      return nullptr;
    }

    return createGPUEngineByType(deviceContext);
#endif
  }

#ifdef VIENNACORE_COMPILE_GPU
  std::shared_ptr<DeviceContext> getOrCreateDeviceContext() const {
    auto deviceContext = DeviceContext::getContextFromRegistry(gpuDeviceId_);
    if (deviceContext) {
      return deviceContext;
    }

    VIENNACORE_LOG_INFO("Auto-generating GPU device context.");

    deviceContext =
        DeviceContext::createContext(VIENNACORE_KERNELS_PATH, gpuDeviceId_);
    if (!deviceContext) {
      VIENNACORE_LOG_ERROR("Failed to create GPU device context.");
    }

    return deviceContext;
  }

  std::unique_ptr<FluxEngine<NumericType, D>>
  createGPUEngineByType(std::shared_ptr<DeviceContext> deviceContext) const {
    switch (fluxEngineType_) {
    case FluxEngineType::GPU_DISK:
      return std::make_unique<GPUDiskEngine<NumericType, D>>(deviceContext);

    case FluxEngineType::GPU_TRIANGLE:
      return std::make_unique<GPUTriangleEngine<NumericType, D>>(deviceContext);

    case FluxEngineType::GPU_LINE:
      if constexpr (D == 3) {
        VIENNACORE_LOG_WARNING("GPU-Line flux engine not supported in 3D. "
                               "Fallback to GPU-Triangle engine.");
        return std::make_unique<GPUTriangleEngine<NumericType, D>>(
            deviceContext);
      }
      return std::make_unique<GPULineEngine<NumericType, D>>(deviceContext);

    default:
      VIENNACORE_LOG_ERROR("Unsupported flux engine type.");
      return nullptr;
    }
  }
#endif

public:
  bool checkInputUpdateContext() {
    if (!context_.domain) {
      VIENNACORE_LOG_ERROR("No domain passed to Process.");
      return false;
    }

    if (context_.domain->getLevelSets().empty()) {
      VIENNACORE_LOG_ERROR("No level sets in domain.");
      return false;
    }

    if (!context_.model) {
      VIENNACORE_LOG_ERROR("No process model passed to Process.");
      return false;
    }

    // Update context with current state
    context_.updateFlags();
    context_.printFlags();

    // Auto-select engine type
    if (fluxEngineType_ == FluxEngineType::AUTO) {
      if (gpuAvailable() && context_.model->hasGPUModel()) {
        // Prefer disks if boundary is periodic in any direction
        if (context_.flags.domainHasPeriodicBoundaries) {
          fluxEngineType_ = FluxEngineType::GPU_DISK;
        } else {
          fluxEngineType_ = FluxEngineType::GPU_TRIANGLE;
        }
      } else {
        fluxEngineType_ = FluxEngineType::CPU_DISK;
      }
      VIENNACORE_LOG_DEBUG("Auto-selected flux engine type: " +
                           to_string(fluxEngineType_));
    }

#ifdef VIENNACORE_COMPILE_GPU
    if (fluxEngineType_ == FluxEngineType::GPU_TRIANGLE ||
        fluxEngineType_ == FluxEngineType::GPU_DISK ||
        fluxEngineType_ == FluxEngineType::GPU_LINE) {

      // Ensure process model has GPU implementation
      auto model =
          std::dynamic_pointer_cast<gpu::ProcessModelGPU<NumericType, D>>(
              context_.model);
      if (!model) {
        // Retry with GPU model if available
        auto deviceContext =
            getOrCreateDeviceContext(); // Ensure context exists
        auto gpuModel = context_.model->getGPUModel();
        if (gpuModel) {
          VIENNACORE_LOG_DEBUG("Switching to GPU-compatible process model.");
          context_.model = gpuModel;
        } else {
          VIENNACORE_LOG_WARNING(
              "No GPU implementation available for this process model.");
          return false;
        }
      }
    }
#endif

    return true;
  }

#ifdef VIENNACORE_COMPILE_GPU
public:
  void setDeviceId(unsigned id) { gpuDeviceId_ = id; }

private:
  unsigned int gpuDeviceId_ = 0;
#endif
};

PS_PRECOMPILE_PRECISION_DIMENSION(Process)

} // namespace viennaps