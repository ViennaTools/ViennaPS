#pragma once

#include "psProcessModel.hpp"
#include "psProcessParams.hpp"
#include "psTranslationField.hpp"
#include "psUnits.hpp"
#include "psUtils.hpp"

#include <lsAdvect.hpp>
#include <lsCalculateVisibilities.hpp>
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
  ProcessBase() {}
  ProcessBase(DomainType passedDomain) : domain_(passedDomain) {}
  // Constructor for a process with a pre-configured process model.
  ProcessBase(DomainType passedDomain,
              SmartPointer<ProcessModel<NumericType, D>> passedProcessModel,
              const NumericType passedDuration = 0.)
      : domain_(passedDomain), model_(passedProcessModel),
        processDuration_(passedDuration) {}

  // Set the process model. This can be either a pre-configured process model or
  // a custom process model. A custom process model must interface the
  // psProcessModel class.
  void setProcessModel(
      SmartPointer<ProcessModel<NumericType, D>> passedProcessModel) {
    model_ = passedProcessModel;
  }

  // Set the process domain.
  void setDomain(DomainType passedDomain) { domain_ = passedDomain; }

  /* ----- Process parameters ----- */

  // Set the duration of the process.
  void setProcessDuration(NumericType passedDuration) {
    processDuration_ = passedDuration;
  }

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
    initRayTracer();
    model_->initialize(domain_, 0.);
    const auto name = model_->getProcessName().value_or("default");
    const auto logLevel = Logger::getLogLevel();

    // Generate disk mesh from domain
    diskMesh_ = SmartPointer<viennals::Mesh<NumericType>>::New();
    viennals::ToDiskMesh<NumericType, D> meshConverter(diskMesh_);
    for (auto dom : domain_->getLevelSets()) {
      meshConverter.insertNextLevelSet(dom);
    }
    meshConverter.apply();
    auto numPoints = diskMesh_->getNodes().size();
    const auto &materialIds =
        *diskMesh_->getCellData().getScalarData("MaterialIds");

    setRayTracerGeometry();

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
      coverageInitIterations(useProcessParams);
    }

    auto fluxes = calculateFluxes(useCoverages, useProcessParams);
    mergeScalarData(diskMesh_->getCellData(), fluxes);
    auto surfaceData = model_->getSurfaceModel()->getSurfaceData();
    if (surfaceData)
      mergeScalarData(diskMesh_->getCellData(), surfaceData);

    return diskMesh_;
  }

  // Run the process.
  void apply() {
    /* ---------- Check input --------- */
    if (!domain_) {
      Logger::getInstance().addWarning("No domain passed to Process.").print();
      return;
    }

    if (domain_->getLevelSets().empty()) {
      Logger::getInstance().addWarning("No level sets in domain.").print();
      return;
    }

    if (!model_) {
      Logger::getInstance()
          .addWarning("No process model passed to Process.")
          .print();
      return;
    }

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

    double remainingTime = processDuration_;
    const NumericType gridDelta = domain_->getGrid().getGridDelta();

    diskMesh_ = SmartPointer<viennals::Mesh<NumericType>>::New();
    auto translator = SmartPointer<TranslatorType>::New();
    viennals::ToDiskMesh<NumericType, D> meshConverter(diskMesh_);
    meshConverter.setTranslator(translator);
    if (domain_->getMaterialMap() &&
        domain_->getMaterialMap()->size() == domain_->getLevelSets().size()) {
      meshConverter.setMaterialMap(domain_->getMaterialMap()->getMaterialMap());
    }

    auto transField = SmartPointer<TranslationField<NumericType, D>>::New(
        model_->getVelocityField(), domain_->getMaterialMap());
    transField->setTranslator(translator);

    viennals::Advect<NumericType, D> advectionKernel;
    advectionKernel.setVelocityField(transField);
    advectionKernel.setIntegrationScheme(advectionParams_.integrationScheme);
    advectionKernel.setTimeStepRatio(advectionParams_.timeStepRatio);
    advectionKernel.setSaveAdvectionVelocities(advectionParams_.velocityOutput);
    advectionKernel.setDissipationAlpha(advectionParams_.dissipationAlpha);
    advectionKernel.setIgnoreVoids(advectionParams_.ignoreVoids);
    advectionKernel.setCheckDissipation(advectionParams_.checkDissipation);
    // normals vectors are only necessary for analytical velocity fields
    if (model_->getVelocityField()->getTranslationFieldOptions() > 0)
      advectionKernel.setCalculateNormalVectors(false);

    for (auto dom : domain_->getLevelSets()) {
      meshConverter.insertNextLevelSet(dom);
      advectionKernel.insertNextLevelSet(dom);
    }

    // Check if the process model uses ray tracing
    const bool useRayTracing = !model_->getParticleTypes().empty();
    if (useRayTracing) {
      initRayTracer();
    }

    // Determine whether advection callback is used
    const bool useAdvectionCallback = model_->getAdvectionCallback() != nullptr;
    if (useAdvectionCallback) {
      model_->getAdvectionCallback()->setDomain(domain_);
    }

    // Determine whether there are process parameters used in ray tracing
    model_->getSurfaceModel()->initializeProcessParameters();
    const bool useProcessParams =
        model_->getSurfaceModel()->getProcessParameters() != nullptr;

    if (useProcessParams)
      Logger::getInstance().addInfo("Using process parameters.").print();
    if (useAdvectionCallback)
      Logger::getInstance().addInfo("Using advection callback.").print();

    bool useCoverages = false;

    // Initialize coverages
    meshConverter.apply();
    auto numPoints = diskMesh_->getNodes().size();
    model_->getSurfaceModel()->initializeSurfaceData(numPoints);
    if (!coveragesInitialized_)
      model_->getSurfaceModel()->initializeCoverages(numPoints);
    auto coverages = model_->getSurfaceModel()->getCoverages();
    if (coverages != nullptr) {
      useCoverages = true;
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

      auto fluxes = SmartPointer<viennals::PointData<NumericType>>::New();
      meshConverter.apply();
      auto materialIds = *diskMesh_->getCellData().getScalarData("MaterialIds");
      auto points = diskMesh_->getNodes();

      // rate calculation by top-down ray tracing
      if (useRayTracing) {
        rtTimer.start();

        setRayTracerGeometry();
        fluxes = calculateFluxes(useCoverages, useProcessParams);

        rtTimer.finish();
        Logger::getInstance()
            .addTiming("Top-down flux calculation", rtTimer)
            .print();
      }

      // update coverages and calculate coverage delta metric
      if (useCoverages) {
        coverages = model_->getSurfaceModel()->getCoverages();
        auto prevStepCoverages =
            SmartPointer<viennals::PointData<NumericType>>::New(*coverages);

        // update coverages in the model
        model_->getSurfaceModel()->updateCoverages(fluxes, materialIds);

        if (coverageDeltaThreshold_ > 0) {
          auto metric =
              this->calculateCoverageDeltaMetric(coverages, prevStepCoverages);
          while (!this->checkCoveragesConvergence(metric)) {
            Logger::getInstance()
                .addInfo("Coverages did not converge. Repeating flux "
                         "calculation.")
                .print();

            prevStepCoverages =
                SmartPointer<viennals::PointData<NumericType>>::New(*coverages);

            rtTimer.start();
            fluxes = calculateFluxes(useCoverages, useProcessParams);
            rtTimer.finish();
            model_->getSurfaceModel()->updateCoverages(fluxes, materialIds);

            coverages = model_->getSurfaceModel()->getCoverages();
            metric = this->calculateCoverageDeltaMetric(coverages,
                                                        prevStepCoverages);
            if (logLevel >= 5) {
              for (auto val : metric) {
                covMetricFile << val << ";";
              }
              covMetricFile << "\n";
            }

            Logger::getInstance()
                .addTiming("Top-down flux calculation", rtTimer)
                .print();
          }
        }
      }

      // calculate velocities from fluxes
      auto velocities = model_->getSurfaceModel()->calculateVelocities(
          fluxes, points, materialIds);

      // prepare velocity field
      model_->getVelocityField()->prepare(domain_, velocities,
                                          processDuration_ - remainingTime);
      if (model_->getVelocityField()->getTranslationFieldOptions() == 2)
        transField->buildKdTree(points);

      // print debug output
      if (logLevel >= 4) {
        if (velocities)
          diskMesh_->getCellData().insertNextScalarData(*velocities,
                                                        "velocities");
        if (useCoverages) {
          auto coverages = model_->getSurfaceModel()->getCoverages();
          this->mergeScalarData(diskMesh_->getCellData(), coverages);
        }
        auto surfaceData = model_->getSurfaceModel()->getSurfaceData();
        if (surfaceData)
          this->mergeScalarData(diskMesh_->getCellData(), surfaceData);
        this->mergeScalarData(diskMesh_->getCellData(), fluxes);
        this->printDiskMesh(diskMesh_,
                            name + "_" + std::to_string(counter) + ".vtp");
        if (domain_->getCellSet()) {
          domain_->getCellSet()->writeVTU(name + "_cellSet_" +
                                          std::to_string(counter) + ".vtu");
        }
        counter++;
      }

      // apply advection callback
      if (useAdvectionCallback) {
        callbackTimer.start();
        bool continueProcess = model_->getAdvectionCallback()->applyPreAdvect(
            processDuration_ - remainingTime);
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
        this->moveCoveragesToTopLS(translator,
                                   model_->getSurfaceModel()->getCoverages());
      advTimer.start();
      advectionKernel.apply();
      advTimer.finish();
      Logger::getInstance().addTiming("Surface advection", advTimer).print();

      if (advectionParams_.velocityOutput) {
        auto lsMesh = SmartPointer<viennals::Mesh<NumericType>>::New();
        viennals::ToMesh<NumericType, D>(domain_->getLevelSets().back(), lsMesh)
            .apply();
        viennals::VTKWriter<NumericType>(
            lsMesh, "ls_velocities_" + std::to_string(lsVelCounter++) + ".vtp")
            .apply();
      }

      // update the translator to retrieve the correct coverages from the LS
      meshConverter.apply();
      if (useCoverages)
        this->updateCoveragesFromAdvectedSurface(
            translator, model_->getSurfaceModel()->getCoverages());

      // apply advection callback
      if (useAdvectionCallback) {
        callbackTimer.start();
        bool continueProcess = model_->getAdvectionCallback()->applyPostAdvect(
            advectionKernel.getAdvectedTime());
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
               << processDuration_ << " "
               << units::Time::getInstance().toShortString();
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
    if (useRayTracing) {
      Logger::getInstance()
          .addTiming("Top-down flux calculation total time",
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
    if (logLevel >= 5)
      covMetricFile.close();
  }

protected:
  static void printDiskMesh(SmartPointer<viennals::Mesh<NumericType>> mesh,
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
      SmartPointer<TranslatorType> translator,
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
      SmartPointer<TranslatorType> translator,
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

  static std::vector<NumericType> calculateCoverageDeltaMetric(
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
    Timer timer;
    Logger::getInstance().addInfo("Using coverages.").print();
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
      timer.start();
      Logger::getInstance().addInfo("Initializing coverages ... ").print();
      // debug output
      if (logLevel >= 5 && !covMetricFile.is_open())
        covMetricFile.open(name + "_covMetric.txt");

      for (unsigned iteration = 0; iteration < maxIterations_; ++iteration) {
        // We need additional signal handling when running the C++ code from
        // the Python bindings to allow interrupts in the Python scripts
#ifdef VIENNAPS_PYTHON_BUILD
        if (PyErr_CheckSignals() != 0)
          throw pybind11::error_already_set();
#endif
        // save current coverages to compare with the new ones
        auto prevStepCoverages =
            SmartPointer<viennals::PointData<NumericType>>::New(*coverages);

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
          auto surfaceData = model_->getSurfaceModel()->getSurfaceData();
          if (surfaceData)
            mergeScalarData(diskMesh_->getCellData(), surfaceData);
          printDiskMesh(diskMesh_, name + "_covInit_" +
                                       std::to_string(iteration) + ".vtp");

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

          // log metric to file
          if (logLevel >= 5) {
            for (auto val : metric) {
              covMetricFile << val << ";";
            }
            covMetricFile << "\n";
          }
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

      if (logLevel >= 5)
        covMetricFile.close();

      timer.finish();
      Logger::getInstance().addTiming("Coverage initialization", timer).print();
    } else {
      Logger::getInstance().addInfo("Coverages already initialized.").print();
    }
  }

  // Implementation specific functions
  virtual bool checkInput() = 0;

  virtual void initRayTracer() = 0;

  virtual void setRayTracerGeometry() = 0;

  virtual SmartPointer<viennals::PointData<NumericType>>
  calculateFluxes(const bool useCoverages, const bool useProcessParams) = 0;

protected:
  DomainType domain_;
  SmartPointer<ProcessModel<NumericType, D>> model_;
  NumericType processDuration_;
  NumericType processTime_ = 0.;
  unsigned maxIterations_ = std::numeric_limits<unsigned>::max();
  NumericType coverageDeltaThreshold_ = 0.;
  bool coveragesInitialized_ = false;

  AdvectionParameters<NumericType> advectionParams_;
  RayTracingParameters<NumericType, D> rayTracingParams_;

  SmartPointer<viennals::Mesh<NumericType>> diskMesh_ = nullptr;
  std::fstream covMetricFile;
};

} // namespace viennaps