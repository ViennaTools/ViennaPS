#pragma once

#include "psAdvectionHandler.hpp"
#include "psCoverageManager.hpp"
#include "psProcessStrategy.hpp"

namespace viennaps {

template <typename NumericType, int D>
class AnalyticProcessStrategy : public ProcessStrategy<NumericType, D> {
private:
  AdvectionHandler<NumericType, D> advectionHandler_;
  viennacore::Timer<> callbackTimer_;

public:
  DEFINE_CLASS_NAME(AnalyticProcessStrategy)

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

  bool canHandle(const ProcessContext<NumericType, D> &context) const override {
    return context.processDuration > 0.0 && !context.flags.isGeometric &&
           context.model->getVelocityField() != nullptr;
  }

private:
  ProcessResult validateContext(const ProcessContext<NumericType, D> &context) {
    if (!context.model->getVelocityField()) {
      Logger::getInstance()
          .addWarning("No velocity field passed to Process.")
          .print();
      return ProcessResult::INVALID_INPUT;
    }

    return ProcessResult::SUCCESS;
  }

  ProcessResult setupProcess(ProcessContext<NumericType, D> &context) {
    // Initialize advection handler
    if (auto result = advectionHandler_.initialize(context);
        result != ProcessResult::SUCCESS) {
      return result;
    }

    return ProcessResult::SUCCESS;
  }

  ProcessResult executeProcessingLoop(ProcessContext<NumericType, D> &context) {
    Timer processTimer;
    processTimer.start();

    while (context.processTime < context.processDuration) {
// Check for user interruption
#ifdef VIENNATOOLS_PYTHON_BUILD
      if (PyErr_CheckSignals() != 0)
        return ProcessResult::USER_INTERRUPTED;
#endif

      // Process one time step
      if (auto result = processTimeStep(context);
          result != ProcessResult::SUCCESS) {
        return result;
      }
    }

    // Finalize process
    context.model->finalize(context.domain, context.processTime);

    processTimer.finish();
    logProcessingTimes(context, processTimer);

    return ProcessResult::SUCCESS;
  }

  ProcessResult processTimeStep(ProcessContext<NumericType, D> &context) {
    // 1. Prepare advection
    advectionHandler_.prepareAdvection(context);

    // 2. Apply advection callbacks (pre)
    if (context.flags.useAdvectionCallback) {
      if (!applyPreAdvectionCallback(context)) {
        return ProcessResult::EARLY_TERMINATION;
      }
    }

    // 3. Perform advection
    auto advectionResult = advectionHandler_.performAdvection(context);
    if (advectionResult.first != ProcessResult::SUCCESS) {
      return advectionResult.first;
    }

    // 4. Update process time
    context.processTime += advectionResult.second;

    // 5. Apply advection callbacks (post)
    if (context.flags.useAdvectionCallback) {
      if (!applyPostAdvectionCallback(context)) {
        return ProcessResult::EARLY_TERMINATION;
      }
    }

    return ProcessResult::SUCCESS;
  }

  bool applyPreAdvectionCallback(ProcessContext<NumericType, D> &context) {
    callbackTimer_.start();
    bool result = context.model->getAdvectionCallback()->applyPreAdvect(
        context.processTime);
    callbackTimer_.finish();
    return result;
  }

  bool applyPostAdvectionCallback(ProcessContext<NumericType, D> &context) {
    callbackTimer_.start();
    bool result = context.model->getAdvectionCallback()->applyPostAdvect(
        context.processTime);
    callbackTimer_.finish();
    return result;
  }

  void logProcessingTimes(const ProcessContext<NumericType, D> &context,
                          const viennacore::Timer<> &processTimer) {
    Logger::getInstance()
        .addTiming("\nProcess " + context.getProcessName(),
                   processTimer.currentDuration * 1e-9)
        .addTiming("Surface advection total time",
                   advectionHandler_.getTimer().totalDuration * 1e-9,
                   processTimer.totalDuration * 1e-9)
        .print();
    if (context.flags.useAdvectionCallback) {
      Logger::getInstance()
          .addTiming("Advection callback total time",
                     callbackTimer_.totalDuration * 1e-9,
                     processTimer.totalDuration * 1e-9)
          .print();
    }
  }
};

} // namespace viennaps