#pragma once

#include "../psUnits.hpp"
#include "psAdvectionHandler.hpp"
#include "psCoverageManager.hpp"
#include "psFluxEngine.hpp"
#include "psProcessStrategy.hpp"

namespace viennaps {

template <typename NumericType, int D>
class ALPStrategy final : public ProcessStrategy<NumericType, D> {
  using TranslatorType = std::unordered_map<unsigned long, unsigned long>;

  AdvectionHandler<NumericType, D> advectionHandler_;
  CoverageManager<NumericType, D> coverageManager_;
  std::unique_ptr<FluxEngine<NumericType, D>> fluxEngine_;

  viennals::ToDiskMesh<NumericType, D> meshGenerator_;
  SmartPointer<TranslatorType> translator_ = nullptr;
  SmartPointer<KDTree<NumericType, Vec3D<NumericType>>> kdTree_ = nullptr;

public:
  DEFINE_CLASS_NAME(ALPStrategy)

  ALPStrategy() = default;
  explicit ALPStrategy(std::unique_ptr<FluxEngine<NumericType, D>> fluxEngine)
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
    return context.flags.isALP;
  }

  bool requiresFluxEngine() const override {
    if (!fluxEngine_)
      return true;
    return false;
  }

  void
  setFluxEngine(std::unique_ptr<FluxEngine<NumericType, D>> engine) override {
    fluxEngine_ = std::move(engine);
  }

private:
  static ProcessResult
  validateContext(const ProcessContext<NumericType, D> &context) {
    if (!context.model->getSurfaceModel()) {
      VIENNACORE_LOG_ERROR("No surface model passed to Atomic Layer Process.");
      return ProcessResult::INVALID_INPUT;
    }

    if (!context.model->getVelocityField()) {
      VIENNACORE_LOG_ERROR("No velocity field passed to Atomic Layer Process.");
      return ProcessResult::INVALID_INPUT;
    }

    return ProcessResult::SUCCESS;
  }

  ProcessResult setupProcess(ProcessContext<NumericType, D> &context) {
    // Initialize disk mesh generator
    context.diskMesh = viennals::Mesh<NumericType>::New();
    meshGenerator_.clearLevelSets();
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

    if (!translator_)
      translator_ = SmartPointer<TranslatorType>::New();
    meshGenerator_.setTranslator(translator_);
    meshGenerator_.apply();
    context.model->getSurfaceModel()->initializeSurfaceData(
        context.diskMesh->nodes.size());

    // Initialize translation field. Converts points ids from level set points
    // to surface points
    const int translationMethod = context.needsExtendedVelocities() ? 2 : 1;
    context.translationField =
        SmartPointer<TranslationField<NumericType, D>>::New(
            context.model->getVelocityField(), context.domain->getMaterialMap(),
            translationMethod);

    if (translationMethod == 1) {
      context.translationField->setTranslator(translator_);
    } else if (translationMethod == 2) {
      kdTree_ = SmartPointer<KDTree<NumericType, Vec3D<NumericType>>>::New();
      context.translationField->setKdTree(kdTree_);
    }

    // Initialize advection handler
    PROCESS_CHECK(advectionHandler_.initialize(context));
    advectionHandler_.disableSingleStep();
    advectionHandler_.setAdvectionTime(1.0); // always advect for one time unit

    if (context.flags.useAdvectionCallback) {
      VIENNACORE_LOG_WARNING("Advection callbacks are not supported in ALP.");
    }

    // Initialize flux engine
    PROCESS_CHECK(fluxEngine_->checkInput(context));
    PROCESS_CHECK(fluxEngine_->initialize(context));

    if (Logger::hasDebug()) {
      // debug output
      std::stringstream ss;
      ss << "Atomic Layer Process: " << context.getProcessName() << "\n"
         << "Grid Delta: " << context.domain->getGridDelta() << "\n"
         << "Atomic Layer Process Parameters: "
         << context.atomicLayerParams.toMetaDataString() << "\n"
         << "Advection Parameters: "
         << context.advectionParams.toMetaDataString() << "\n"
         << "Ray Tracing Parameters: "
         << context.rayTracingParams.toMetaDataString() << "\n";
      Logger::getInstance().addDebug(ss.str()).print();
    }

    return ProcessResult::SUCCESS;
  }

  ProcessResult executeProcessingLoop(ProcessContext<NumericType, D> &context) {
    Timer processTimer;
    processTimer.start();

    const auto numCycles = context.atomicLayerParams.numCycles;
    const auto pulseTime = context.atomicLayerParams.pulseTime;
    const auto purgePulseTime = context.atomicLayerParams.purgePulseTime;

    for (int cycle = 0; cycle < numCycles; ++cycle) {
      VIENNACORE_LOG_INFO("Cycle: " + std::to_string(cycle + 1) + "/" +
                          std::to_string(numCycles));

      // Prepare advection (expand level set based on integration scheme)
      advectionHandler_.prepareAdvection(context);

      updateState(context);
      PROCESS_CHECK(fluxEngine_->updateSurface(context));
      if (!coverageManager_.initializeCoverages(context)) {
        VIENNACORE_LOG_WARNING("No coverages found in ALP model.");
        return ProcessResult::INVALID_INPUT;
      }

      double time = 0.;
      unsigned pulseIteration = 0;
      while (std::fabs(time - pulseTime) > 1e-6) {
#ifdef VIENNATOOLS_PYTHON_BUILD
        // Check for user interruption
        if (PyErr_CheckSignals() != 0)
          return ProcessResult::USER_INTERRUPTED;
#endif

        // Calculate fluxes
        auto fluxes = SmartPointer<viennals::PointData<NumericType>>::New();
        PROCESS_CHECK(fluxEngine_->calculateFluxes(context, fluxes));

        // Update coverages
        updateCoverages(context, fluxes);

        outputIntermediateResults(context, fluxes, pulseIteration);

        time += context.atomicLayerParams.coverageTimeStep;
        pulseIteration++;

        if (Logger::hasInfo()) {
          std::stringstream stream;
          stream << std::fixed << std::setprecision(4) << "Pulse time: " << time
                 << " / " << pulseTime << " " << units::Time::toShortString();
          Logger::getInstance().addInfo(stream.str()).print();
        }
      }

      if (purgePulseTime > 0.) {
        VIENNACORE_LOG_WARNING(
            "Purge pulses are not implemented yet. Skipping purge step.");
        /// TODO: Implement purge step
      }

      // Calculate velocities in model
      auto fluxes = SmartPointer<viennals::PointData<NumericType>>::New();
      PROCESS_CHECK(fluxEngine_->calculateFluxes(context, fluxes));
      auto velocities = calculateVelocities(context, fluxes);
      context.model->getVelocityField()->prepare(context.domain, velocities,
                                                 0.);

      // We don't move the coverages during the advection step, because they are
      // re-initialized each cycle

      // print intermediate output
      if (Logger::hasIntermediate()) {
        const auto name = context.getProcessName();
        context.diskMesh->getCellData().insertNextScalarData(*velocities,
                                                             "velocities");
        viennals::VTKWriter<NumericType>(
            context.diskMesh,
            name + "_" + std::to_string(context.currentIteration) + ".vtp")
            .apply();
      }

      // Perform advection, updates processTime, reduces level set to width 1
      PROCESS_CHECK(advectionHandler_.performAdvection(context));

      ++context.currentIteration;
    }

    // Finalize process
    context.model->finalize(context.domain, context.processTime);

    processTimer.finish();
    logProcessingTimes(context, processTimer);

    // Add metadata to domain
    if (static_cast<int>(context.domain->getMetaDataLevel()) > 1) {
      context.domain->addMetaData(context.atomicLayerParams.toMetaData());
    }
    if (static_cast<int>(context.domain->getMetaDataLevel()) > 2) {
      context.domain->addMetaData(context.advectionParams.toMetaData());
      context.domain->addMetaData(context.rayTracingParams.toMetaData());
    }

    return ProcessResult::SUCCESS;
  }

  void outputIntermediateResults(
      ProcessContext<NumericType, D> &context,
      SmartPointer<viennals::PointData<NumericType>> const &fluxes,
      const unsigned pulseIteration) {
    if (Logger::hasIntermediate()) {
      auto const name = context.getProcessName();
      auto surfaceModel = context.model->getSurfaceModel();
      mergeScalarData(context.diskMesh->getCellData(),
                      surfaceModel->getCoverages());
      mergeScalarData(context.diskMesh->getCellData(), fluxes);
      if (auto surfaceData = surfaceModel->getSurfaceData())
        mergeScalarData(context.diskMesh->getCellData(), surfaceData);
      viennals::VTKWriter<NumericType>(
          context.diskMesh, context.intermediateOutputPath + name + "_pulse_" +
                                std::to_string(context.currentIteration) + "_" +
                                std::to_string(pulseIteration) + ".vtp")
          .apply();
    }
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
    if (!Logger::hasTiming())
      return;
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
  }
};

} // namespace viennaps