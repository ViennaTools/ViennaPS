#pragma once

#include "../psUnits.hpp"
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

  // Timers
  viennacore::Timer<> callbackTimer_;

public:
  DEFINE_CLASS_NAME(FluxProcessStrategy)

  FluxProcessStrategy(std::unique_ptr<FluxEngine<NumericType, D>> fluxEngine)
      : fluxEngine_(std::move(fluxEngine)) {}

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
           !context.flags.isAnalytic && context.flags.useFluxEngine &&
           !context.flags.isALP;
  }

  ProcessResult calculateFlux(ProcessContext<NumericType, D> &context) {
    // Validate required components
    PROCESS_CHECK(validateContext(context));

    // Setup phase
    PROCESS_CHECK(setupProcess(context));

    if (context.flags.useCoverages) {
      coverageInitIterations(context);
    }

    updateState(context);
    PROCESS_CHECK(fluxEngine_->updateSurface(context));

    // Calculate fluxes only
    auto fluxes = SmartPointer<viennals::PointData<NumericType>>::New();
    PROCESS_CHECK(fluxEngine_->calculateFluxes(context, fluxes));

    mergeScalarData(context.diskMesh->getCellData(), fluxes);

    return ProcessResult::SUCCESS;
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
    // Initialize disk mesh generator
    context.diskMesh = viennals::Mesh<NumericType>::New();
    meshGenerator_.setMesh(context.diskMesh);
    for (auto &dom : context.domain->getLevelSets()) {
      meshGenerator_.insertNextLevelSet(dom);
    }
    if (context.domain->getMaterialMap() &&
        context.domain->getMaterialMap()->size() ==
            context.domain->getLevelSets().size()) {
      meshGenerator_.setMaterialMap(
          context.domain->getMaterialMap()->getMaterialMap());
    }

    // Try to initialize coverages
    meshGenerator_.apply();
    if (!context.coverageParams.initialized) {
      if (coverageManager_.initializeCoverages(context)) {
        context.flags.useCoverages = true;
        Logger::getInstance().addInfo("Using coverages.").print();
      }

      if (!translator_)
        translator_ = SmartPointer<TranslatorType>::New();
      meshGenerator_.setTranslator(translator_);
    } else {
      Logger::getInstance().addInfo("Coverages already initialized.").print();
    }
    context.model->getSurfaceModel()->initializeSurfaceData(
        context.diskMesh->nodes.size());

    // Initialize translation field. Converts points ids from level set points
    // to surface points
    context.translationField =
        SmartPointer<TranslationField<NumericType, D>>::New(
            context.model->getVelocityField(),
            context.domain->getMaterialMap());

    auto translationMethod = context.translationField->getTranslationMethod();
    if (translationMethod == 0) {
      Logger::getInstance()
          .addWarning(
              "Translation method can not be 0 for flux-based processes.")
          .print();
      return ProcessResult::INVALID_INPUT;
    } else if (translationMethod == 1) {
      if (!translator_)
        translator_ = SmartPointer<TranslatorType>::New();
      context.translationField->setTranslator(translator_);
    } else if (translationMethod == 2) {
      kdTree_ = SmartPointer<KDTree<NumericType, Vec3D<NumericType>>>::New();
      context.translationField->setKdTree(kdTree_);
    }

    // Initialize advection handler
    PROCESS_CHECK(advectionHandler_.initialize(context));

    // Initialize flux engine
    PROCESS_CHECK(fluxEngine_->checkInput(context));
    PROCESS_CHECK(fluxEngine_->initialize(context));

    return ProcessResult::SUCCESS;
  }

  ProcessResult executeProcessingLoop(ProcessContext<NumericType, D> &context) {
    Timer processTimer;
    processTimer.start();

    if (context.flags.useCoverages && !context.coverageParams.initialized) {
      coverageInitIterations(context);
    }

    context.resetTime();

    while (context.processTime < context.processDuration) {
#ifdef VIENNATOOLS_PYTHON_BUILD
      // Check for user interruption
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
      context.domain->addMetaData(context.rayTracingParams.toMetaData());
      if (context.flags.useCoverages) {
        context.domain->addMetaData(context.coverageParams.toMetaData());
      }
    }

    return ProcessResult::SUCCESS;
  }

  ProcessResult processTimeStep(ProcessContext<NumericType, D> &context) {
    // Prepare advection (expand level set based on integration scheme)
    advectionHandler_.prepareAdvection(context);

    // Update surface for flux calculation
    updateState(context);
    PROCESS_CHECK(fluxEngine_->updateSurface(context));

    // Calculate fluxes
    auto fluxes = SmartPointer<viennals::PointData<NumericType>>::New();
    PROCESS_CHECK(fluxEngine_->calculateFluxes(context, fluxes));

    // Update coverages if needed
    if (context.flags.useCoverages) {
      updateCoverages(context, fluxes);
    }

    // Calculate velocities in model
    auto velocities = calculateVelocities(context, fluxes);
    context.model->getVelocityField()->prepare(context.domain, velocities,
                                               context.processTime);

    // Apply advection callbacks (pre)
    if (context.flags.useAdvectionCallback) {
      if (!applyPreAdvectionCallback(context)) {
        return ProcessResult::EARLY_TERMINATION;
      }
    }

    // Move coverages to level set to propagate them during advection
    if (context.flags.useCoverages) {
      PROCESS_CHECK(
          advectionHandler_.copyCoveragesToLevelSet(context, translator_));
    }

    outputIntermediateResults(context, velocities, fluxes);

    // Perform advection, updates processTime, reduces level set to width 1
    PROCESS_CHECK(advectionHandler_.performAdvection(context));

    // Update coverages from advected surface
    if (context.flags.useCoverages) {
      meshGenerator_.apply();
      PROCESS_CHECK(advectionHandler_.updateCoveragesFromAdvectedSurface(
          context, translator_));
    }

    // Apply advection callbacks (post)
    if (context.flags.useAdvectionCallback) {
      if (!applyPostAdvectionCallback(context)) {
        return ProcessResult::EARLY_TERMINATION;
      }
    }

    if (Logger::getLogLevel() >= 2) {
      std::stringstream stream;
      stream << std::fixed << std::setprecision(4)
             << "Process time: " << context.processTime << " / "
             << context.processDuration << " " << units::Time::toShortString();
      Logger::getInstance().addInfo(stream.str()).print();
    }

    return ProcessResult::SUCCESS;
  }

  void outputIntermediateResults(
      ProcessContext<NumericType, D> &context,
      SmartPointer<std::vector<NumericType>> &velocities,
      SmartPointer<viennals::PointData<NumericType>> &fluxes) {
    if (Logger::getLogLevel() >= 3) {
      auto const name = context.getProcessName();
      auto surfaceModel = context.model->getSurfaceModel();
      context.diskMesh->getCellData().insertNextScalarData(*velocities,
                                                           "velocities");
      if (context.flags.useCoverages) {
        mergeScalarData(context.diskMesh->getCellData(),
                        surfaceModel->getCoverages());
      }
      if (auto surfaceData = surfaceModel->getSurfaceData())
        mergeScalarData(context.diskMesh->getCellData(), surfaceData);
      mergeScalarData(context.diskMesh->getCellData(), fluxes);
      viennals::VTKWriter<NumericType>(
          context.diskMesh,
          name + "_" + std::to_string(context.currentIteration) + ".vtp")
          .apply();
      if (context.domain->getCellSet()) {
        context.domain->getCellSet()->writeVTU(
            name + "_cellSet_" + std::to_string(context.currentIteration) +
            ".vtu");
      }
    }
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

  SmartPointer<std::vector<NumericType>>
  calculateVelocities(const ProcessContext<NumericType, D> &context,
                      SmartPointer<viennals::PointData<NumericType>> &fluxes) {
    auto const &points = context.diskMesh->getNodes();
    assert(points.size() > 0);
    auto const &materialIds =
        *context.diskMesh->getCellData().getScalarData("MaterialIds");
    return context.model->getSurfaceModel()->calculateVelocities(fluxes, points,
                                                                 materialIds);
  }

  void updateCoverages(ProcessContext<NumericType, D> &context,
                       SmartPointer<viennals::PointData<NumericType>> &fluxes) {
    auto surfaceModel = context.model->getSurfaceModel();
    assert(surfaceModel != nullptr);
    assert(surfaceModel->getCoverages() != nullptr);

    assert(context.diskMesh != nullptr);
    assert(context.diskMesh->getCellData().getScalarData("MaterialIds") !=
           nullptr);
    auto const &materialIds =
        *context.diskMesh->getCellData().getScalarData("MaterialIds");

    surfaceModel->updateCoverages(fluxes, materialIds);
  }

  void updateState(ProcessContext<NumericType, D> &context) {
    meshGenerator_.apply();

    auto const translationMethod =
        context.translationField->getTranslationMethod();
    if (translationMethod == 2) {
      kdTree_->setPoints(context.diskMesh->getNodes());
      kdTree_->build();
    }
  }

  void coverageInitIterations(ProcessContext<NumericType, D> &context) {

    const auto name = context.getProcessName();
    auto &maxIterations = context.coverageParams.maxIterations;
    if (maxIterations == std::numeric_limits<unsigned>::max() &&
        context.coverageParams.coverageDeltaThreshold == 0.) {
      maxIterations = 10;
      Logger::getInstance()
          .addWarning("No coverage initialization parameters set. Using " +
                      std::to_string(maxIterations) +
                      " initialization iterations.")
          .print();
    }

    Timer timer;
    timer.start();
    Logger::getInstance().addInfo("Initializing coverages ... ").print();

    fluxEngine_->updateSurface(context);
    for (unsigned iteration = 0; iteration < maxIterations; ++iteration) {
#ifdef VIENNATOOLS_PYTHON_BUILD
      // Check for user interruption
      if (PyErr_CheckSignals() != 0)
        throw pybind11::error_already_set();
#endif
      // save current coverages to compare with the new ones
      coverageManager_.saveCoverages(context);

      auto fluxes = SmartPointer<viennals::PointData<NumericType>>::New();
      fluxEngine_->calculateFluxes(context, fluxes);

      updateCoverages(context, fluxes);

      if (Logger::getLogLevel() >= 3) {
        auto coverages = context.model->getSurfaceModel()->getCoverages();
        mergeScalarData(context.diskMesh->getCellData(), coverages);
        mergeScalarData(context.diskMesh->getCellData(), fluxes);
        if (auto surfaceData =
                context.model->getSurfaceModel()->getSurfaceData())
          mergeScalarData(context.diskMesh->getCellData(), surfaceData);
        viennals::VTKWriter(context.diskMesh, name + "_covInit_" +
                                                  std::to_string(iteration) +
                                                  ".vtp")
            .apply();
      }

      if (coverageManager_.checkCoveragesConvergence(context)) {
        Logger::getInstance()
            .addInfo("Coverages converged after " +
                     std::to_string(iteration + 1) + " iterations.")
            .print();
        break;
      }
    }
    context.coverageParams.initialized = true;
    timer.finish();
    Logger::getInstance().addTiming("Coverage initialization", timer).print();
  }

  static void
  mergeScalarData(viennals::PointData<NumericType> &scalarData,
                  SmartPointer<viennals::PointData<NumericType>> dataToInsert) {
    int numScalarData = dataToInsert->getScalarDataSize();
    for (int i = 0; i < numScalarData; i++) {
      scalarData.insertReplaceScalarData(*dataToInsert->getScalarData(i),
                                         dataToInsert->getScalarDataLabel(i));
    }
  }

  void logProcessingTimes(const ProcessContext<NumericType, D> &context,
                          const viennacore::Timer<> &processTimer) {
    Logger::getInstance()
        .addTiming("\nProcess " + context.getProcessName(),
                   processTimer.currentDuration * 1e-9)
        .addTiming("Surface advection total time",
                   advectionHandler_.getTimer().totalDuration * 1e-9,
                   processTimer.totalDuration * 1e-9)
        .addTiming("Flux engine total time",
                   fluxEngine_->getTimer().totalDuration * 1e-9,
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