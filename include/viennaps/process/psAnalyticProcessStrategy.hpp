#pragma once

#include "psAdvectionHandler.hpp"
#include "psCoverageManager.hpp"
#include "psFluxCalculator.hpp"
#include "psProcessStrategy.hpp"

namespace viennaps {

template <typename NumericType, int D>
class AnalyticProcessStrategy : public ProcessStrategy<NumericType, D> {
private:
  std::unique_ptr<AdvectionHandler<NumericType, D>> advectionHandler_;

public:
  AnalyticProcessStrategy(
      std::unique_ptr<AdvectionHandler<NumericType, D>> advectionHandler)
      : advectionHandler_(std::move(advectionHandler)) {}

  ProcessResult execute(ProcessContext<NumericType, D> &context) override {
    // Validate required components
    if (auto result = validateContext(context);
        result != ProcessResult::SUCCESS) {
      return result;
    }

    // Setup phase
    if (auto result = setupProcess(context); result != ProcessResult::SUCCESS) {
      return result;
    }

    // Main processing loop
    return executeProcessingLoop(context);
  }

  std::string getStrategyName() const override {
    return "AnalyticProcessStrategy";
  }

  bool canHandle(const ProcessContext<NumericType, D> &context) const override {
    return context.processDuration > 0.0 && !context.flags.isGeometric &&
           context.model->getSurfaceModel() != nullptr &&
           context.model->getVelocityField() != nullptr;
  }

private:
  ProcessResult validateContext(const ProcessContext<NumericType, D> &context) {
    if (!context.model->getSurfaceModel()) {
      Logger::getInstance()
          .addError("No surface model passed to Process.")
          .print();
      return ProcessResult::INVALID_INPUT;
    }

    if (!context.model->getVelocityField()) {
      Logger::getInstance()
          .addError("No velocity field passed to Process.")
          .print();
      return ProcessResult::INVALID_INPUT;
    }

    return ProcessResult::SUCCESS;
  }

  ProcessResult setupProcess(ProcessContext<NumericType, D> &context) {
    // Initialize advection handler
    if (auto result = advectionHandler_->initialize(context);
        result != ProcessResult::SUCCESS) {
      return result;
    }

    // Initialize flux calculator if needed
    if (context.flags.useFluxEngine) {
      if (auto result = fluxCalculator_->initialize(context);
          result != ProcessResult::SUCCESS) {
        return result;
      }
    }

    return ProcessResult::SUCCESS;
  }

  ProcessResult executeProcessingLoop(ProcessContext<NumericType, D> &context) {
    Timer processTimer;
    processTimer.start();

    context.remainingTime = context.processDuration;
    size_t counter = 0;

    while (context.remainingTime > 0.) {
// Check for user interruption
#ifdef VIENNATOOLS_PYTHON_BUILD
      if (PyErr_CheckSignals() != 0)
        return ProcessResult::USER_INTERRUPTED;
#endif

      // Process one time step
      auto result = processTimeStep(context, counter);
      if (result != ProcessResult::SUCCESS) {
        return result;
      }

      counter++;
    }

    // Finalize process
    context.processTime = context.processDuration - context.remainingTime;
    context.model->finalize(context.domain, context.processTime);

    processTimer.finish();
    logProcessingTimes(context, processTimer);

    return ProcessResult::SUCCESS;
  }

  ProcessResult processTimeStep(ProcessContext<NumericType, D> &context,
                                size_t counter) {
    // 1. Prepare advection
    advectionHandler_->prepareAdvection(context);

    // 2. Apply advection callbacks (pre)
    if (context.flags.useAdvectionCallback) {
      if (!applyPreAdvectionCallback(context)) {
        return ProcessResult::EARLY_TERMINATION;
      }
    }

    // 3. Perform advection
    auto advectionResult = advectionHandler_->performAdvection(context);
    if (advectionResult.first != ProcessResult::SUCCESS) {
      return advectionResult.first;
    }

    // 4. Apply advection callbacks (post)
    if (context.flags.useAdvectionCallback) {
      if (!applyPostAdvectionCallback(context, advectionResult.second)) {
        return ProcessResult::EARLY_TERMINATION;
      }
    }

    // 5. Update remaining time
    context.remainingTime -= advectionResult.second;

    // 6. Output intermediate results if needed
    outputIntermediateResults(context, counter);

    return ProcessResult::SUCCESS;
  }

  // Additional helper methods...
};

} // namespace viennaps