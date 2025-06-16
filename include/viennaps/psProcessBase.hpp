#pragma once

#include "psProcessModelBase.hpp"
#include "psProcessParams.hpp"
#include "psTranslationField.hpp"
#include "psUnits.hpp"

#include <lsAdvect.hpp>
#include <lsDomain.hpp>
#include <lsMesh.hpp>
#include <lsToDiskMesh.hpp>
#include <lsVTKWriter.hpp>

namespace viennaps {

using namespace viennacore;

/// This class server as the main process tool, applying a user- or pre-defined
/// process model to a domain. Depending on the user inputs surface advection, a
/// single callback function or a geometric advection is applied.
template <typename NumericType, int D> class ProcessBase {
  using TranslatorType = std::unordered_map<unsigned long, unsigned long>;
  using DomainType = SmartPointer<Domain<NumericType, D>>;

public:
  ProcessBase() = default;
  ProcessBase(DomainType domain) : domain_(domain) {}
  // Constructor for a process with a pre-configured process model.
  ProcessBase(DomainType domain,
              SmartPointer<ProcessModelBase<NumericType, D>> processModel,
              const NumericType duration = 0.)
      : domain_(domain), model_(processModel), processDuration_(duration) {}
  virtual ~ProcessBase() { assert(!covMetricFile.is_open()); }

  // Set the process domain.
  void setDomain(DomainType domain) { domain_ = domain; }

  /* ----- Process parameters ----- */

  // Set the duration of the process.
  void setProcessDuration(NumericType duration) { processDuration_ = duration; }

  // Returns the duration of the recently run process. This duration can
  // sometimes slightly vary from the set process duration, due to the maximum
  // time step according to the CFL condition.
  NumericType getProcessDuration() const { return processTime_; }

  /* ----- Ray tracing parameters ----- */

  // Set the number of iterations to initialize the coverages.
  void setMaxCoverageInitIterations(unsigned maxIt) { maxIterations_ = maxIt; }

  // Set the threshold for the coverage delta metric to reach convergence.
  void setCoverageDeltaThreshold(NumericType threshold) {
    coverageDeltaThreshold_ = threshold;
  }

  // Set the source direction, where the rays should be traced from.
  void setSourceDirection(viennaray::TraceDirection passedDirection) {
    rayTracingParams_.sourceDirection = passedDirection;
  }

  // Specify the number of rays to be traced for each particle throughout the
  // process. The total count of rays is the product of this number and the
  // number of points in the process geometry.
  void setNumberOfRaysPerPoint(unsigned numRays) {
    rayTracingParams_.raysPerPoint = numRays;
  }

  // Enable flux smoothing. The flux at each surface point, calculated
  // by the ray tracer, is averaged over the surface point neighbors.
  void enableFluxSmoothing() { rayTracingParams_.smoothingNeighbors = 1; }

  // Disable flux smoothing.
  void disableFluxSmoothing() { rayTracingParams_.smoothingNeighbors = 0; }

  void enableFluxBoundaries() {
    rayTracingParams_.ignoreFluxBoundaries = false;
  }

  // Ignore boundary conditions during the flux calculation.
  void disableFluxBoundaries() {
    rayTracingParams_.ignoreFluxBoundaries = true;
  }

  // Enable the use of random seeds for ray tracing. This is useful to
  // prevent the formation of artifacts in the flux calculation.
  void enableRandomSeeds() { rayTracingParams_.useRandomSeeds = true; }

  // Disable the use of random seeds for ray tracing.
  void disableRandomSeeds() { rayTracingParams_.useRandomSeeds = false; }

  void setRayTracingParameters(
      const RayTracingParameters<NumericType, D> &passedRayTracingParams) {
    rayTracingParams_ = passedRayTracingParams;
  }

  auto &getRayTracingParameters() { return rayTracingParams_; }

  /* ----- Advection parameters ----- */

  // Set the integration scheme for solving the level-set equation.
  // Possible integration schemes are specified in
  // viennals::IntegrationSchemeEnum.
  void setIntegrationScheme(IntegrationScheme passedIntegrationScheme) {
    advectionParams_.integrationScheme = passedIntegrationScheme;
  }

  // Enable the output of the advection velocities on the level-set mesh.
  void enableAdvectionVelocityOutput() {
    advectionParams_.velocityOutput = true;
  }

  // Disable the output of the advection velocities on the level-set mesh.
  void disableAdvectionVelocityOutput() {
    advectionParams_.velocityOutput = false;
  }

  // Set the CFL (Courant-Friedrichs-Levy) condition to use during surface
  // advection in the level-set. The CFL condition defines the maximum distance
  // a surface is allowed to move in a single advection step. It MUST be below
  // 0.5 to guarantee numerical stability. Defaults to 0.4999.
  void setTimeStepRatio(NumericType cfl) {
    advectionParams_.timeStepRatio = cfl;
  }

  void setAdvectionParameters(
      const AdvectionParameters<NumericType> &passedAdvectionParams) {
    advectionParams_ = passedAdvectionParams;
  }

  auto &getAdvectionParameters() { return advectionParams_; }

  /* ----- Process execution ----- */

  // A single flux calculation is performed on the domain surface. The result is
  // stored as point data on the nodes of the mesh.
  SmartPointer<viennals::Mesh<NumericType>> calculateFlux() {

    if (!checkModelAndDomain())
      return nullptr;

    model_->initialize(domain_, 0.);
    const auto name = model_->getProcessName().value_or("default");

    if (!model_->useFluxEngine()) {
      Logger::getInstance()
          .addWarning("Process model '" + name + "' does not use flux engine.")
          .print();
      return nullptr;
    }

    if (!checkInput())
      return nullptr;

    initFluxEngine();
    const auto logLevel = Logger::getLogLevel();

    // Generate disk mesh from domain
    diskMesh_ = viennals::Mesh<NumericType>::New();
    viennals::ToDiskMesh<NumericType, D> meshGenerator(diskMesh_);
    if (domain_->getMaterialMap() &&
        domain_->getMaterialMap()->size() == domain_->getLevelSets().size()) {
      meshGenerator.setMaterialMap(domain_->getMaterialMap()->getMaterialMap());
    }
    for (auto ls : domain_->getLevelSets()) {
      meshGenerator.insertNextLevelSet(ls);
    }
    meshGenerator.apply();
    auto numPoints = diskMesh_->getNodes().size();
    const auto &materialIds =
        *diskMesh_->getCellData().getScalarData("MaterialIds");

    setFluxEngineGeometry();

    translationField_ = SmartPointer<TranslationField<NumericType, D>>::New(
        model_->getVelocityField(), domain_->getMaterialMap());

    const bool useProcessParams =
        model_->getSurfaceModel()->getProcessParameters() != nullptr;
    bool useCoverages = false;

    model_->getSurfaceModel()->initializeSurfaceData(numPoints);
    // Check if coverages are used
    if (!coveragesInitialized_)
      model_->getSurfaceModel()->initializeCoverages(numPoints);
    auto coverages = model_->getSurfaceModel()->getCoverages();

    if (coverages != nullptr) {
      useCoverages = true;
      Logger::getInstance().addInfo("Using coverages.").print();

      // debug output
      if (logLevel >= 5)
        covMetricFile.open(name + "_covMetric.txt");

      coverageInitIterations(useProcessParams);

      if (logLevel >= 5)
        covMetricFile.close();
    }

    auto fluxes = calculateFluxes(useCoverages, useProcessParams);
    mergeScalarData(diskMesh_->getCellData(), fluxes);
    if (auto surfaceData = model_->getSurfaceModel()->getSurfaceData())
      mergeScalarData(diskMesh_->getCellData(), surfaceData);

    return diskMesh_;
  }

  // Run the process.
  void apply() {
    /* ---------- Check input --------- */
    if (!checkModelAndDomain())
      return;

    model_->initialize(domain_, processDuration_);
    const auto name = model_->getProcessName().value_or("default");

    if (model_->getGeometricModel()) {
      model_->getGeometricModel()->setDomain(domain_);
      Logger::getInstance().addInfo("Applying geometric model...").print();
      model_->getGeometricModel()->apply();
      return;
    }

    if (processDuration_ == 0.) {
      // apply only advection callback
      if (model_->getAdvectionCallback()) {
        model_->getAdvectionCallback()->setDomain(domain_);
        model_->getAdvectionCallback()->applyPreAdvect(0);
      } else {
        Logger::getInstance()
            .addWarning("No advection callback passed to Process.")
            .print();
      }
      return;
    }

    if (!model_->getSurfaceModel()) {
      Logger::getInstance()
          .addWarning("No surface model passed to Process.")
          .print();
      return;
    }

    if (!model_->getVelocityField()) {
      Logger::getInstance()
          .addWarning("No velocity field passed to Process.")
          .print();
      return;
    }

    // check implementation specific input
    if (!checkInput())
      return;

    /* ------ Process Setup ------ */
    const unsigned int logLevel = Logger::getLogLevel();
    Timer processTimer;
    processTimer.start();

    auto surfaceModel = model_->getSurfaceModel();
    auto velocityField = model_->getVelocityField();
    auto advectionCallback = model_->getAdvectionCallback();
    double remainingTime = processDuration_;
    const NumericType gridDelta = domain_->getGrid().getGridDelta();

    diskMesh_ = viennals::Mesh<NumericType>::New();
    auto translator = SmartPointer<TranslatorType>::New();
    viennals::ToDiskMesh<NumericType, D> meshGenerator(diskMesh_);
    meshGenerator.setTranslator(translator);
    if (domain_->getMaterialMap() &&
        domain_->getMaterialMap()->size() == domain_->getLevelSets().size()) {
      meshGenerator.setMaterialMap(domain_->getMaterialMap()->getMaterialMap());
    }

    translationField_ = SmartPointer<TranslationField<NumericType, D>>::New(
        velocityField, domain_->getMaterialMap());
    translationField_->setTranslator(translator);

    viennals::Advect<NumericType, D> advectionKernel;
    advectionKernel.setVelocityField(translationField_);
    advectionKernel.setIntegrationScheme(advectionParams_.integrationScheme);
    advectionKernel.setTimeStepRatio(advectionParams_.timeStepRatio);
    advectionKernel.setSaveAdvectionVelocities(advectionParams_.velocityOutput);
    advectionKernel.setDissipationAlpha(advectionParams_.dissipationAlpha);
    advectionKernel.setIgnoreVoids(advectionParams_.ignoreVoids);
    advectionKernel.setCheckDissipation(advectionParams_.checkDissipation);
    // normals vectors are only necessary for analytical velocity fields
    if (velocityField->getTranslationFieldOptions() > 0)
      advectionKernel.setCalculateNormalVectors(false);

    for (auto dom : domain_->getLevelSets()) {
      meshGenerator.insertNextLevelSet(dom);
      advectionKernel.insertNextLevelSet(dom);
    }

    // Check if the process model uses ray tracing
    const bool useFluxEngine = model_->useFluxEngine();
    if (useFluxEngine) {
      initFluxEngine();
    }

    // Determine whether advection callback is used
    const bool useAdvectionCallback = advectionCallback != nullptr;
    if (useAdvectionCallback) {
      advectionCallback->setDomain(domain_);
    }

    // Determine whether there are process parameters used in ray tracing
    surfaceModel->initializeProcessParameters();
    const bool useProcessParams =
        surfaceModel->getProcessParameters() != nullptr;

    if (useProcessParams)
      Logger::getInstance().addInfo("Using process parameters.").print();
    if (useAdvectionCallback)
      Logger::getInstance().addInfo("Using advection callback.").print();

    bool useCoverages = false;

    // Initialize coverages
    meshGenerator.apply();
    auto numPoints = diskMesh_->getNodes().size();
    surfaceModel->initializeSurfaceData(numPoints);
    if (!coveragesInitialized_)
      surfaceModel->initializeCoverages(numPoints);
    auto coverages = surfaceModel->getCoverages();
    if (coverages != nullptr) {
      Logger::getInstance().addInfo("Using coverages.").print();
      useCoverages = true;

      // debug output
      if (logLevel >= 5)
        covMetricFile.open(name + "_covMetric.txt");

      setFluxEngineGeometry();
      coverageInitIterations(useProcessParams);
    }

    double previousTimeStep = 0.;
    size_t counter = 0;
    unsigned lsVelCounter = 0;
    Timer rtTimer;
    Timer callbackTimer;
    Timer advTimer;
    while (remainingTime > 0.) {
      // We need additional signal handling when running the C++ code from the
      // Python bindings to allow interrupts in the Python scripts
#ifdef VIENNAPS_PYTHON_BUILD
      if (PyErr_CheckSignals() != 0)
        throw pybind11::error_already_set();
#endif
      // Expand LS based on the integration scheme
      advectionKernel.prepareLS();
      model_->initialize(domain_, remainingTime);

      auto fluxes = viennals::PointData<NumericType>::New();
      meshGenerator.apply();
      auto materialIds = *diskMesh_->getCellData().getScalarData("MaterialIds");
      auto points = diskMesh_->getNodes();

      // rate calculation by implementation specific flux engine
      if (useFluxEngine) {
        rtTimer.start();

        setFluxEngineGeometry();
        fluxes = calculateFluxes(useCoverages, useProcessParams);

        rtTimer.finish();
        Logger::getInstance().addTiming("Flux calculation", rtTimer).print();
      }

      // update coverages and calculate coverage delta metric
      if (useCoverages) {
        coverages = surfaceModel->getCoverages();
        auto prevStepCoverages =
            viennals::PointData<NumericType>::New(*coverages);

        // update coverages in the model
        surfaceModel->updateCoverages(fluxes, materialIds);

        if (coverageDeltaThreshold_ > 0) {
          auto metric =
              this->calculateCoverageDeltaMetric(coverages, prevStepCoverages);
          while (!this->checkCoveragesConvergence(metric)) {
            Logger::getInstance()
                .addDebug("Coverages did not converge within threshold. "
                          "Repeating flux "
                          "calculation.")
                .print();

            prevStepCoverages =
                viennals::PointData<NumericType>::New(*coverages);

            rtTimer.start();
            fluxes = calculateFluxes(useCoverages, useProcessParams);
            rtTimer.finish();
            surfaceModel->updateCoverages(fluxes, materialIds);

            coverages = surfaceModel->getCoverages();
            metric = this->calculateCoverageDeltaMetric(coverages,
                                                        prevStepCoverages);

            Logger::getInstance()
                .addTiming("Top-down flux calculation", rtTimer)
                .print();
          }
        }
      }

      // calculate velocities from fluxes
      auto velocities =
          surfaceModel->calculateVelocities(fluxes, points, materialIds);

      // prepare velocity field
      velocityField->prepare(domain_, velocities,
                             processDuration_ - remainingTime);
      if (velocityField->getTranslationFieldOptions() == 2)
        translationField_->buildKdTree(points);

      // print debug output
      if (logLevel >= 4) {
        if (velocities)
          diskMesh_->getCellData().insertNextScalarData(*velocities,
                                                        "velocities");
        if (useCoverages) {
          mergeScalarData(diskMesh_->getCellData(),
                          surfaceModel->getCoverages());
        }
        if (auto surfaceData = surfaceModel->getSurfaceData())
          mergeScalarData(diskMesh_->getCellData(), surfaceData);
        mergeScalarData(diskMesh_->getCellData(), fluxes);
        saveDiskMesh(diskMesh_, name + "_" + std::to_string(counter) + ".vtp");
        if (domain_->getCellSet()) {
          domain_->getCellSet()->writeVTU(name + "_cellSet_" +
                                          std::to_string(counter) + ".vtu");
        }
        counter++;
      }

      // apply advection callback
      if (useAdvectionCallback) {
        callbackTimer.start();
        bool continueProcess =
            advectionCallback->applyPreAdvect(processDuration_ - remainingTime);
        callbackTimer.finish();
        Logger::getInstance()
            .addTiming("Advection callback pre-advect", callbackTimer)
            .print();

        if (!continueProcess) {
          Logger::getInstance()
              .addInfo("Process stopped early by AdvectionCallback during "
                       "`preAdvect`.")
              .print();
          break;
        }
      }

      // adjust time step near end
      if (remainingTime - previousTimeStep < 0.) {
        advectionKernel.setAdvectionTime(remainingTime);
      }

      // move coverages to LS, so they are moved with the advection step
      if (useCoverages)
        this->moveCoveragesToTopLS(translator, surfaceModel->getCoverages());
      advTimer.start();
      advectionKernel.apply();
      advTimer.finish();
      Logger::getInstance().addTiming("Surface advection", advTimer).print();

      if (advectionParams_.velocityOutput) {
        auto lsMesh = viennals::Mesh<NumericType>::New();
        viennals::ToMesh<NumericType, D>(domain_->getLevelSets().back(), lsMesh)
            .apply();
        viennals::VTKWriter<NumericType>(
            lsMesh, "ls_velocities_" + std::to_string(lsVelCounter++) + ".vtp")
            .apply();
      }

      // update the translator to retrieve the correct coverages from the LS
      meshGenerator.apply();
      if (useCoverages)
        this->updateCoveragesFromAdvectedSurface(translator,
                                                 surfaceModel->getCoverages());

      // apply advection callback
      if (useAdvectionCallback) {
        callbackTimer.start();
        auto advectedTime = advectionKernel.getAdvectedTime();
        advectedTime = advectedTime == std::numeric_limits<double>::max()
                           ? remainingTime
                           : advectedTime;
        bool continueProcess = advectionCallback->applyPostAdvect(advectedTime);
        callbackTimer.finish();
        Logger::getInstance()
            .addTiming("Advection callback post-advect", callbackTimer)
            .print();
        if (!continueProcess) {
          Logger::getInstance()
              .addInfo("Process stopped early by AdvectionCallback during "
                       "`postAdvect`.")
              .print();
          break;
        }
      }

      previousTimeStep = advectionKernel.getAdvectedTime();
      if (previousTimeStep == std::numeric_limits<NumericType>::max()) {
        Logger::getInstance()
            .addInfo("Process halted: Surface velocities are zero across the "
                     "entire surface.")
            .print();
        remainingTime = 0.;
        break;
      }
      remainingTime -= previousTimeStep;

      if (Logger::getLogLevel() >= 2) {
        std::stringstream stream;
        stream << std::fixed << std::setprecision(4)
               << "Process time: " << processDuration_ - remainingTime << " / "
               << processDuration_ << " " << units::Time::toShortString();
        Logger::getInstance().addInfo(stream.str()).print();
      }
    }

    processTime_ = processDuration_ - remainingTime;
    processTimer.finish();

    Logger::getInstance()
        .addTiming("\nProcess " + name, processTimer)
        .addTiming("Surface advection total time",
                   advTimer.totalDuration * 1e-9,
                   processTimer.totalDuration * 1e-9)
        .print();
    if (useFluxEngine) {
      Logger::getInstance()
          .addTiming("Flux calculation total time",
                     rtTimer.totalDuration * 1e-9,
                     processTimer.totalDuration * 1e-9)
          .print();
    }
    if (useAdvectionCallback) {
      Logger::getInstance()
          .addTiming("Advection callback total time",
                     callbackTimer.totalDuration * 1e-9,
                     processTimer.totalDuration * 1e-9)
          .print();
    }
    model_->reset();
    if (useCoverages && logLevel >= 5)
      covMetricFile.close();
  }

protected:
  bool checkModelAndDomain() const {
    if (!domain_) {
      Logger::getInstance().addWarning("No domain passed to Process.").print();
      return false;
    }

    if (domain_->getLevelSets().empty()) {
      Logger::getInstance().addWarning("No level sets in domain.").print();
      return false;
    }

    if (!model_) {
      Logger::getInstance()
          .addWarning("No process model passed to Process.")
          .print();
      return false;
    }

    return true;
  }

  static void saveDiskMesh(SmartPointer<viennals::Mesh<NumericType>> mesh,
                           std::string name) {
    viennals::VTKWriter<NumericType>(mesh, std::move(name)).apply();
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

  void moveCoveragesToTopLS(
      SmartPointer<TranslatorType> const &translator,
      SmartPointer<viennals::PointData<NumericType>> coverages) {
    auto topLS = domain_->getLevelSets().back();
    for (size_t i = 0; i < coverages->getScalarDataSize(); i++) {
      auto covName = coverages->getScalarDataLabel(i);
      std::vector<NumericType> levelSetData(topLS->getNumberOfPoints(), 0);
      auto cov = coverages->getScalarData(covName);
      for (const auto iter : *translator.get()) {
        levelSetData[iter.first] = cov->at(iter.second);
      }
      if (auto data = topLS->getPointData().getScalarData(covName, true);
          data != nullptr) {
        *data = std::move(levelSetData);
      } else {
        topLS->getPointData().insertNextScalarData(std::move(levelSetData),
                                                   covName);
      }
    }
  }

  void updateCoveragesFromAdvectedSurface(
      SmartPointer<TranslatorType> const &translator,
      SmartPointer<viennals::PointData<NumericType>> coverages) const {
    auto topLS = domain_->getLevelSets().back();
    for (size_t i = 0; i < coverages->getScalarDataSize(); i++) {
      auto covName = coverages->getScalarDataLabel(i);
      auto levelSetData = topLS->getPointData().getScalarData(covName);
      auto covData = coverages->getScalarData(covName);
      covData->resize(translator->size());
      for (const auto it : *translator.get()) {
        covData->at(it.second) = levelSetData->at(it.first);
      }
    }
  }

  std::vector<NumericType> calculateCoverageDeltaMetric(
      SmartPointer<viennals::PointData<NumericType>> updated,
      SmartPointer<viennals::PointData<NumericType>> previous) {

    assert(updated->getScalarDataSize() == previous->getScalarDataSize());
    std::vector<NumericType> delta(updated->getScalarDataSize(), 0.);

#pragma omp parallel for
    for (int i = 0; i < updated->getScalarDataSize(); i++) {
      auto label = updated->getScalarDataLabel(i);
      auto updatedData = updated->getScalarData(label);
      auto previousData = previous->getScalarData(label);
      for (size_t j = 0; j < updatedData->size(); j++) {
        auto diff = updatedData->at(j) - previousData->at(j);
        delta[i] += diff * diff;
      }

      delta[i] /= updatedData->size();
    }

    logMetric(delta);
    return delta;
  }

  bool
  checkCoveragesConvergence(const std::vector<NumericType> &deltaMetric) const {
    for (auto val : deltaMetric) {
      if (val > coverageDeltaThreshold_)
        return false;
    }
    return true;
  }

  void coverageInitIterations(const bool useProcessParams) {
    auto coverages = model_->getSurfaceModel()->getCoverages();
    const auto name = model_->getProcessName().value_or("default");
    const auto logLevel = Logger::getLogLevel();

    if (maxIterations_ == std::numeric_limits<unsigned>::max() &&
        coverageDeltaThreshold_ == 0.) {
      maxIterations_ = 10;
      Logger::getInstance()
          .addWarning("No coverage initialization parameters set. Using " +
                      std::to_string(maxIterations_) +
                      " initialization iterations.")
          .print();
    }

    if (!coveragesInitialized_) {
      Timer timer;
      timer.start();
      Logger::getInstance().addInfo("Initializing coverages ... ").print();

      for (unsigned iteration = 0; iteration < maxIterations_; ++iteration) {
        // We need additional signal handling when running the C++ code from
        // the Python bindings to allow interrupts in the Python scripts
#ifdef VIENNAPS_PYTHON_BUILD
        if (PyErr_CheckSignals() != 0)
          throw pybind11::error_already_set();
#endif
        // save current coverages to compare with the new ones
        assert(coverages != nullptr);
        auto prevStepCoverages =
            viennals::PointData<NumericType>::New(*coverages);

        auto fluxes = calculateFluxes(true, useProcessParams);

        // update coverages in the model
        assert(diskMesh_ != nullptr);
        auto const &materialIds =
            *diskMesh_->getCellData().getScalarData("MaterialIds");
        model_->getSurfaceModel()->updateCoverages(fluxes, materialIds);

        // metric to check for convergence
        coverages = model_->getSurfaceModel()->getCoverages();
        auto metric =
            calculateCoverageDeltaMetric(coverages, prevStepCoverages);

        if (logLevel >= 3) {
          mergeScalarData(diskMesh_->getCellData(), coverages);
          mergeScalarData(diskMesh_->getCellData(), fluxes);
          if (auto surfaceData = model_->getSurfaceModel()->getSurfaceData())
            mergeScalarData(diskMesh_->getCellData(), surfaceData);
          saveDiskMesh(diskMesh_,
                       name + "_covInit_" + std::to_string(iteration) + ".vtp");

          Logger::getInstance()
              .addInfo("Iteration: " + std::to_string(iteration + 1))
              .print();

          std::stringstream stream;
          stream << std::setprecision(4) << std::fixed;
          stream << "Coverage delta metric: ";
          for (int i = 0; i < coverages->getScalarDataSize(); i++) {
            stream << coverages->getScalarDataLabel(i) << ": " << metric[i]
                   << "\t";
          }
          Logger::getInstance().addInfo(stream.str()).print();
        }

        if (checkCoveragesConvergence(metric)) {
          Logger::getInstance()
              .addInfo("Coverages converged after " +
                       std::to_string(iteration + 1) + " iterations.")
              .print();
          break;
        }
      } // end coverage initialization loop
      coveragesInitialized_ = true;

      timer.finish();
      Logger::getInstance().addTiming("Coverage initialization", timer).print();
    } else {
      Logger::getInstance().addInfo("Coverages already initialized.").print();
    }
  }

  void logMetric(const std::vector<NumericType> &metric) {
    if (Logger::getLogLevel() < 5)
      return;

    assert(covMetricFile.is_open());
    for (auto val : metric) {
      covMetricFile << val << ";";
    }
    covMetricFile << "\n";
  }

  // Implementation specific functions (to be implemented by derived classes,
  // currently CPU or GPU Process)
  virtual bool checkInput() = 0;

  virtual void initFluxEngine() = 0;

  virtual void setFluxEngineGeometry() = 0;

  virtual SmartPointer<viennals::PointData<NumericType>>
  calculateFluxes(const bool useCoverages, const bool useProcessParams) = 0;

protected:
  DomainType domain_;
  SmartPointer<ProcessModelBase<NumericType, D>> model_;
  NumericType processDuration_;
  NumericType processTime_ = 0.;
  unsigned maxIterations_ = std::numeric_limits<unsigned>::max();
  NumericType coverageDeltaThreshold_ = 0.;
  bool coveragesInitialized_ = false;

  AdvectionParameters<NumericType> advectionParams_;
  RayTracingParameters<NumericType, D> rayTracingParams_;

  SmartPointer<viennals::Mesh<NumericType>> diskMesh_ = nullptr;
  SmartPointer<TranslationField<NumericType, D>> translationField_ = nullptr;
  std::ofstream covMetricFile;
};

} // namespace viennaps