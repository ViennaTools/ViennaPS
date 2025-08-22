#pragma once

#include "psProcessContext.hpp"

// Process strategies
#include "psAnalyticProcessStrategy.hpp"
#include "psCallbackOnlyStrategy.hpp"
#include "psFluxProcessStrategy.hpp"
#include "psGeometricProcessStrategy.hpp"

// Flux engines
#include "psCPUDiskEngine.hpp"

#include "../psProcessModelBase.hpp"

namespace viennaps {

template <typename NumericType, int D> class Process {
private:
  ProcessContext<NumericType, D> context_;
  std::vector<std::unique_ptr<ProcessStrategy<NumericType, D>>> strategies_;
  FluxEngineType fluxEngineType_ = FluxEngineType::CPU_Disk;

public:
  Process() { initializeStrategies(); }
  Process(SmartPointer<Domain<NumericType, D>> domain,
          SmartPointer<ProcessModelBase<NumericType, D>> model,
          NumericType processDuration = 0.)
      : context_{domain, model, processDuration} {
    initializeStrategies();
  }

  void apply() {
    if (!checkModelAndDomain())
      return;

    // Update context with current state
    context_.updateFlags();

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
    case ProcessResult::FAILURE:
      Logger::getInstance().addError("Process failed.").print();
      break;
      // Handle other results...
    }
  }

  // Factory method for creating flux engines (can be overridden by derived
  // classes)
  std::unique_ptr<FluxEngine<NumericType, D>> createFluxEngine() {
    switch (fluxEngineType_) {
    case FluxEngineType::CPU_Disk:
      return std::make_unique<CPUDiskEngine<NumericType, D>>();
    default:
      Logger::getInstance().addError("Unsupported flux engine type.").print();
      return nullptr;
    }
  }

  bool checkModelAndDomain() const {
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

    return true;
  }
};

} // namespace viennaps