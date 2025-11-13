#pragma once

#include "../psUnits.hpp"
#include "psAdvectionHandler.hpp"
#include "psCoverageManager.hpp"
#include "psProcessStrategy.hpp"

namespace viennaps {

template <typename NumericType, int D>
class AnalyticProcessStrategy : public ProcessStrategy<NumericType, D> {
private:
  viennals::ToDiskMesh<NumericType, D> meshConverter_;
  AdvectionHandler<NumericType, D> advectionHandler_;
  viennacore::Timer<> callbackTimer_;

public:
  DEFINE_CLASS_NAME(AnalyticProcessStrategy)

  ProcessResult execute(ProcessContext<NumericType, D> &context) override {
    // Validate required components
    PROCESS_CHECK(validateContext(context));

    // Setup phase
    PROCESS_CHECK(setupProcess(context));

    // Main processing loop
    return executeProcessingLoop(context);
  }

  bool canHandle(const ProcessContext<NumericType, D> &context) const override {
    return context.processDuration > 0.0 && !context.flags.isGeometric &&
           !context.flags.useFluxEngine && context.flags.isAnalytic;
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
    context.model->initialize(context.domain, context.processTime);
    context.translationField =
        SmartPointer<TranslationField<NumericType, D>>::New(
            context.model->getVelocityField(), context.domain->getMaterialMap(),
            0);

    // Initialize advection handler
    PROCESS_CHECK(advectionHandler_.initialize(context));

    if (context.flags.useAdvectionCallback) {
      context.model->getAdvectionCallback()->setDomain(context.domain);
    }

    // Set up disk mesh for intermediate output if requested
    if (Logger::getLogLevel() >=
        static_cast<unsigned>(LogLevel::INTERMEDIATE)) {
      context.diskMesh = viennals::Mesh<NumericType>::New();
      meshConverter_.setMesh(context.diskMesh);
      meshConverter_.clearLevelSets();
      for (auto &dom : context.domain->getLevelSets()) {
        meshConverter_.insertNextLevelSet(dom);
      }
      if (context.domain->getMaterialMap()) {
        meshConverter_.setMaterialMap(
            context.domain->getMaterialMap()->getMaterialMap());
      }
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
      PROCESS_CHECK(processTimeStep(context));

      context.currentIteration++;
    }

    // Finalize process
    context.model->finalize(context.domain, context.processTime);

    processTimer.finish();
    logProcessingTimes(context, processTimer);

    if (static_cast<int>(context.domain->getMetaDataLevel()) > 1) {
      context.domain->addMetaData("ProcessTime", context.processTime);
    }
    if (static_cast<int>(context.domain->getMetaDataLevel()) > 2) {
      context.domain->addMetaData(context.advectionParams.toMetaData());
    }

    return ProcessResult::SUCCESS;
  }

  ProcessResult processTimeStep(ProcessContext<NumericType, D> &context) {
    // Prepare advection based on integration scheme
    // Initialize model
    advectionHandler_.prepareAdvection(context);

    // Apply advection callbacks (pre)
    if (context.flags.useAdvectionCallback) {
      if (!applyPreAdvectionCallback(context)) {
        return ProcessResult::EARLY_TERMINATION;
      }
    }

    // Prepare velocity field (no fluxes for analytic processes)
    context.model->getVelocityField()->prepare(context.domain, nullptr,
                                               context.processTime);

    if (Logger::getLogLevel() >=
        static_cast<unsigned>(LogLevel::INTERMEDIATE)) {
      auto const name = context.getProcessName();
      if (context.domain->getCellSet()) {
        context.domain->getCellSet()->writeVTU(
            name + "_cellSet_" + std::to_string(context.currentIteration) +
            ".vtu");
      }
      meshConverter_.apply();
      viennals::VTKWriter<NumericType>(
          context.diskMesh,
          name + "_" + std::to_string(context.currentIteration) + ".vtp")
          .apply();
    }

    // Perform advection, update processTime
    PROCESS_CHECK(advectionHandler_.performAdvection(context));

    // Apply advection callbacks (post)
    if (context.flags.useAdvectionCallback) {
      if (!applyPostAdvectionCallback(context)) {
        return ProcessResult::EARLY_TERMINATION;
      }
    }

    if (Logger::getLogLevel() >= static_cast<unsigned>(LogLevel::INFO)) {
      std::stringstream stream;
      stream << std::fixed << std::setprecision(4)
             << "Process time: " << context.processTime << " / "
             << context.processDuration << " " << units::Time::toShortString();
      Logger::getInstance().addInfo(stream.str()).print();
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