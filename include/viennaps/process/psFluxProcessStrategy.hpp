#pragma once

#include "psAdvectionHandler.hpp"
#include "psCoverageManager.hpp"
#include "psFluxEngine.hpp"
#include "psProcessStrategy.hpp"

#include <lsToDiskMesh.hpp>

namespace viennaps {

template <typename NumericType, int D>
class FluxProcessStrategy : public ProcessStrategy<NumericType, D> {
  using TranslatorType = std::unordered_map<unsigned long, unsigned long>;

private:
  AdvectionHandler<NumericType, D> advectionHandler_;
  CoverageManager<NumericType, D> coverageManager_;
  std::unique_ptr<FluxEngine<NumericType, D>> fluxEngine_;

  viennals::ToDiskMesh<NumericType, D> meshGenerator_;
  SmartPointer<TranslatorType> translator_ = nullptr;
  SmartPointer<KDTree<NumericType, Vec3D<NumericType>>> kdTree_ = nullptr;

public:
  DEFINE_CLASS_NAME(FluxProcessStrategy)

  FluxProcessStrategy(std::unique_ptr<FluxEngine<NumericType, D>> fluxEngine)
      : fluxEngine_(std::move(fluxEngine)) {}

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
    return ProcessResult::SUCCESS;
    // return executeProcessingLoop(context);
  }

  bool canHandle(const ProcessContext<NumericType, D> &context) const override {
    return context.processDuration > 0.0 && !context.flags.isGeometric &&
           context.flags.useFluxEngine &&
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
    if (auto result = advectionHandler_.initialize(context);
        result != ProcessResult::SUCCESS) {
      return result;
    }

    // Initialize disk mesh generator
    context.diskMesh = viennals::Mesh<NumericType>::New();
    meshGenerator_.setMesh(context.diskMesh);
    for (auto &dom : context.domain->getLevelSets()) {
      meshGenerator_.insertNextLevelSet(dom);
    }

    // Initialize translator
    auto translationField = advectionHandler_.getTranslationField();
    auto translationMethod = translationField->getTranslationMethod();
    if (translationMethod == 0) {
      Logger::getInstance()
          .addError("Translation method can not be 0 for flux-based processes.")
          .print();
      return ProcessResult::INVALID_INPUT;
    } else if (translationMethod == 1) {
      translator_ = SmartPointer<TranslatorType>::New();
      meshGenerator_.setTranslator(translator_);
      translationField->setTranslator(translator_);
    } else if (translationMethod == 2) {
      kdTree_ = SmartPointer<KDTree<NumericType, Vec3D<NumericType>>>::New();
      translationField->setKdTree(kdTree_);
    }

    // Run mesh generator to create disk mesh
    meshGenerator_.apply();

    // Initialize flux engine
    if (auto result = fluxEngine_->checkInput(context);
        result != ProcessResult::SUCCESS) {
      return result;
    }
    if (auto result = fluxEngine_->initialize(context);
        result != ProcessResult::SUCCESS) {
      return result;
    }

    // Try to initialize coverages if needed
    if (coverageManager_.initializeCoverages(context)) {
      Logger::getInstance().addInfo("Using coverages.").print();
      context.flags.useCoverages = true;
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

    // 2. Update surface for flux calculation
    if (auto result = fluxEngine_->updateSurface(context);
        result != ProcessResult::SUCCESS) {
      return result;
    }

    // 3. Calculate fluxes if needed
    SmartPointer<viennals::PointData<NumericType>> fluxes;
    {
      auto result = fluxEngine_->calculateFluxes(context);
      if (result.first != ProcessResult::SUCCESS) {
        return result.first;
      }
      fluxes = result.second;
    }

    // 4. Update coverages if needed
    if (context.flags.useCoverages) {
      auto result = coverageManager_->updateCoverages(context, fluxes);
      if (result != ProcessResult::SUCCESS) {
        return result;
      }
    }

    // 5. Calculate velocities
    auto velocities = calculateVelocities(context, fluxes);

    // 6. Apply advection callbacks (pre)
    if (context.flags.useAdvectionCallback) {
      if (!applyPreAdvectionCallback(context)) {
        return ProcessResult::EARLY_TERMINATION;
      }
    }

    // 7. Perform advection
    auto advectionResult = advectionHandler_->performAdvection(context);
    if (advectionResult.first != ProcessResult::SUCCESS) {
      return advectionResult.first;
    }

    // 8. Apply advection callbacks (post)
    if (context.flags.useAdvectionCallback) {
      if (!applyPostAdvectionCallback(context, advectionResult.second)) {
        return ProcessResult::EARLY_TERMINATION;
      }
    }

    // 9. Update remaining time
    context.remainingTime -= advectionResult.second;

    // 10. Output intermediate results if needed
    outputIntermediateResults(context, counter);

    return ProcessResult::SUCCESS;
  }

  // Additional helper methods...
};

} // namespace viennaps