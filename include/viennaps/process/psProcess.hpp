#pragma once

#include "psProcessContext.hpp"

// Process strategies
#include "psALPStrategy.hpp"
#include "psAnalyticProcessStrategy.hpp"
#include "psCallbackOnlyStrategy.hpp"
#include "psFluxProcessStrategy.hpp"
#include "psGeometricProcessStrategy.hpp"

// Flux engines
#include "psCPUDiskEngine.hpp"
#include "psGPUTriangleEngine.hpp"

namespace viennaps {

template <typename NumericType, int D> class Process {
private:
  ProcessContext<NumericType, D> context_;
  std::vector<std::unique_ptr<ProcessStrategy<NumericType, D>>> strategies_;
  FluxEngineType fluxEngineType_ = FluxEngineType::CPU_DISK;

public:
  Process() = default;
  Process(SmartPointer<Domain<NumericType, D>> domain) : context_{domain} {}
  Process(SmartPointer<Domain<NumericType, D>> domain,
          SmartPointer<ProcessModelBase<NumericType, D>> model,
          NumericType processDuration = 0.)
      : context_{domain, model, processDuration} {}

  void setDomain(SmartPointer<Domain<NumericType, D>> domain) {
    context_.domain = domain;
  }

  void setProcessModel(SmartPointer<ProcessModelBase<NumericType, D>> model) {
    context_.model = model;
  }

  void setProcessDuration(double duration) {
    context_.processDuration = duration;
  }

  void setRayTracingParameters(const RayTracingParameters<D> &params) {
    context_.rayTracingParams = params;
  }

  void setAdvectionParameters(const AdvectionParameters &params) {
    context_.advectionParams = params;
  }

  void setCoverageParameters(const CoverageParameters &params) {
    context_.coverageParams = params;
  }

  void
  setAtomicLayerProcessParameters(const AtomicLayerProcessParameters &params) {
    context_.atomicLayerParams = params;
  }

  void setFluxEngineType(FluxEngineType type) { fluxEngineType_ = type; }

  void apply() {
    if (!checkInput())
      return;

    // Update context with current state
    context_.updateFlags();
    context_.printFlags();

    initializeStrategies();

    // Find appropriate strategy
    auto strategy = findStrategy();
    if (!strategy) {
      Logger::getInstance()
          .addError("No suitable strategy found for process configuration.")
          .print();
      return;
    }
    Logger::getInstance()
        .addDebug("Using strategy: " + std::string(strategy->name()))
        .print();

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
    if (!checkInput())
      return nullptr;

    context_.updateFlags();
    const auto name = context_.getProcessName();
    if (!context_.flags.useFluxEngine) {
      Logger::getInstance()
          .addError("Process model '" + name + "' does not use flux engine.")
          .print();
      return nullptr;
    }

    auto strategy = std::make_unique<FluxProcessStrategy<NumericType, D>>(
        createFluxEngine());
    strategy->calculateFlux(context_);

    return context_.diskMesh;
  }

private:
  void initializeStrategies() {
    // Add strategies in priority order
    strategies_.push_back(
        std::make_unique<GeometricProcessStrategy<NumericType, D>>());
    strategies_.push_back(
        std::make_unique<CallbackOnlyStrategy<NumericType, D>>());
    strategies_.push_back(
        std::make_unique<AnalyticProcessStrategy<NumericType, D>>());
    strategies_.push_back(std::make_unique<FluxProcessStrategy<NumericType, D>>(
        createFluxEngine()));
    strategies_.push_back(
        std::make_unique<ALPStrategy<NumericType, D>>(createFluxEngine()));
  }

  ProcessStrategy<NumericType, D> *findStrategy() {
    for (auto &strategy : strategies_) {
      if (strategy->canHandle(context_)) {
        return strategy.get();
      }
    }
    return nullptr;
  }

  void handleProcessResult(ProcessResult result) {
    switch (result) {
    case ProcessResult::SUCCESS:
      // Handle success
      break;
    case ProcessResult::EARLY_TERMINATION:
      Logger::getInstance().addInfo("Process terminated early.").print();
      break;
    case ProcessResult::USER_INTERRUPTED:
      Logger::getInstance().addInfo("Process interrupted by user.").print();
#ifdef VIENNATOOLS_PYTHON_BUILD
      throw pybind11::error_already_set();
#endif
      break;
    case ProcessResult::INVALID_INPUT:
      Logger::getInstance().addError("Invalid input for process.").print();
      break;
    case ProcessResult::FAILURE:
      Logger::getInstance().addError("Process failed.").print();
      break;
    }
  }

  // Factory method for creating flux engines
  std::unique_ptr<FluxEngine<NumericType, D>> createFluxEngine() {
    switch (fluxEngineType_) {
    case FluxEngineType::CPU_DISK:
      return std::make_unique<CPUDiskEngine<NumericType, D>>();
    case FluxEngineType::GPU_TRIANGLE:
#ifdef VIENNACORE_COMPILE_GPU
      return std::make_unique<GPUTriangleEngine<NumericType, D>>(
          DeviceContext::getContextFromRegistry(gpuDeviceId_));
#else
      Logger::getInstance()
          .addError("GPU support not compiled in ViennaCore.")
          .print();
      return nullptr;
#endif
    default:
      Logger::getInstance().addError("Unsupported flux engine type.").print();
      return nullptr;
    }
  }

  bool checkInput() {
    if (!context_.domain) {
      Logger::getInstance().addError("No domain passed to Process.").print();
      return false;
    }

    if (context_.domain->getLevelSets().empty()) {
      Logger::getInstance().addError("No level sets in domain.").print();
      return false;
    }

    if (!context_.model) {
      Logger::getInstance()
          .addError("No process model passed to Process.")
          .print();
      return false;
    }

#ifdef VIENNACORE_COMPILE_GPU
    if (fluxEngineType_ == FluxEngineType::GPU_TRIANGLE) {
      if (D == 2) {
        Logger::getInstance()
            .addError("GPU-Triangle flux engine not supported in 2D.")
            .print();
        return false;
      }

      auto model =
          std::dynamic_pointer_cast<gpu::ProcessModelGPU<NumericType, D>>(
              context_.model);
      if (!model) {
        auto gpuModel = context_.model->getGPUModel();
        if (gpuModel) {
          Logger::getInstance()
              .addDebug("Switching to GPU-compatible process model.")
              .print();
          context_.model = gpuModel;
        } else {
          Logger::getInstance()
              .addWarning(
                  "No GPU implementation available for this process model.")
              .print();
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

} // namespace viennaps