#pragma once

#include "psProcessContext.hpp"

// Process strategies
#include "psAnalyticProcessStrategy.hpp"
#include "psCallbackOnlyStrategy.hpp"
#include "psFluxProcessStrategy.hpp"
#include "psGeometricProcessStrategy.hpp"

#include "psFluxEngine.hpp"

namespace viennaps {

template <typename NumericType, int D> class Process {
private:
  ProcessContext<NumericType, D> context_;
  std::vector<std::unique_ptr<ProcessStrategy<NumericType, D>>> strategies_;
  FluxEngineType fluxEngineType_ = FluxEngineType::CPU_Disk;

public:
  Process() { initializeStrategies(); }

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

    // Execute strategy
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

    // Surface strategy needs additional components
    auto advectionHandler =
        std::make_unique<AdvectionHandler<NumericType, D>>();
    auto fluxCalculator = createFluxCalculator(); // Factory method
    auto coverageManager = std::make_unique<CoverageManager<NumericType, D>>();

    strategies_.push_back(std::make_unique<FluxProcessStrategy<NumericType, D>>(
        std::move(advectionHandler), std::move(fluxCalculator),
        std::move(coverageManager)));
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
      break;
      // Handle other results...
    }
  }

  // Factory method for creating flux calculators (can be overridden by derived
  // classes)
  std::unique_ptr<FluxEngine<NumericType, D>> createFluxCalculator() {
    switch (fluxEngineType_) {
    case FluxEngineType::CPU_Disk:
      return std::make_unique<CPUDiskEngine<NumericType, D>>();
    default:
      Logger::getInstance().addError("Unsupported flux engine type.").print();
      return nullptr;
    }
  }
};

} // namespace viennaps