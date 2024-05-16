#pragma once

#include "psLogger.hpp"
#include "psProcessModel.hpp"
#include "psTranslationField.hpp"

#include <lsAdvect.hpp>
#include <lsDomain.hpp>
#include <lsMesh.hpp>
#include <lsToDiskMesh.hpp>

#include <rayParticle.hpp>
#include <rayTrace.hpp>

/// This class server as the main process tool, applying a user- or pre-defined
/// process model to a domain. Depending on the user inputs surface advection, a
/// single callback function or a geometric advection is applied.
template <typename NumericType, int D> class psProcess {
  using translatorType = std::unordered_map<unsigned long, unsigned long>;
  using psDomainType = psSmartPointer<psDomain<NumericType, D>>;

public:
  psProcess() {}
  psProcess(psSmartPointer<psDomain<NumericType, D>> passedDomain)
      : domain(passedDomain) {}
  // Constructor for a process with a pre-configured process model.
  template <typename ProcessModelType>
  psProcess(psSmartPointer<psDomain<NumericType, D>> passedDomain,
            psSmartPointer<ProcessModelType> passedProcessModel,
            const NumericType passedDuration = 0.)
      : domain(passedDomain), processDuration(passedDuration) {
    model = std::dynamic_pointer_cast<psProcessModel<NumericType, D>>(
        passedProcessModel);
  }

  // Set the process model. This can be either a pre-configured process model or
  // a custom process model. A custom process model must interface the
  // psProcessModel class.
  template <typename ProcessModelType,
            lsConcepts::IsBaseOf<psProcessModel<NumericType, D>,
                                 ProcessModelType> = lsConcepts::assignable>
  void setProcessModel(psSmartPointer<ProcessModelType> passedProcessModel) {
    model = std::dynamic_pointer_cast<psProcessModel<NumericType, D>>(
        passedProcessModel);
  }

  // Set the process domain.
  void setDomain(psSmartPointer<psDomain<NumericType, D>> passedDomain) {
    domain = passedDomain;
  }

  // Set the source direction, where the rays should be traced from.
  void setSourceDirection(rayTraceDirection passedDirection) {
    sourceDirection = passedDirection;
  }

  // Set the duration of the process.
  void setProcessDuration(NumericType passedDuration) {
    processDuration = passedDuration;
  }

  // Returns the duration of the recently run process. This duration can
  // sometimes slightly vary from the set process duration, due to the maximum
  // time step according to the CFL condition.
  NumericType getProcessDuration() const { return processTime; }

  // Specify the number of rays to be traced for each particle throughout the
  // process. The total count of rays is the product of this number and the
  // number of points in the process geometry.
  void setNumberOfRaysPerPoint(unsigned numRays) { raysPerPoint = numRays; }

  // Set the number of iterations to initialize the coverages.
  void setMaxCoverageInitIterations(unsigned maxIt) { maxIterations = maxIt; }

  /// Enable flux smoothing. The flux at each surface point, calculated
  /// by the ray tracer, is averaged over the surface point neighbors.
  void enableFluxSmoothing() { smoothFlux = true; }

  // Disable flux smoothing.
  void disableFluxSmoothing() { smoothFlux = false; }

  void enableFluxBoundaries() { ignoreFluxBoundaries = false; }

  // Ignore boundary conditions during the flux calculation.
  void disableFluxBoundaries() { ignoreFluxBoundaries = true; }

  // Set the integration scheme for solving the level-set equation.
  // Possible integration schemes are specified in lsIntegrationSchemeEnum.
  void setIntegrationScheme(lsIntegrationSchemeEnum passedIntegrationScheme) {
    integrationScheme = passedIntegrationScheme;
  }

  // Enable the use of random seeds for ray tracing. This is useful to
  // prevent the formation of artifacts in the flux calculation.
  void enableRandomSeeds() { useRandomSeeds_ = true; }

  // Disable the use of random seeds for ray tracing.
  void disableRandomSeeds() { useRandomSeeds_ = false; }

  // Set the CFL (Courant-Friedrichs-Levy) condition to use during surface
  // advection in the level-set. The CFL condition defines the maximum distance
  // a surface is allowed to move in a single advection step. It MUST be below
  // 0.5 to guarantee numerical stability. Defaults to 0.4999.
  void setTimeStepRatio(NumericType cfl) { timeStepRatio = cfl; }

  // Sets the minimum time between printing intermediate results during the
  // process. If this is set to a non-positive value, no intermediate results
  // are printed.
  void setPrintTimeInterval(NumericType passedTime) { printTime = passedTime; }

  // A single flux calculation is performed on the domain surface. The result is
  // stored as point data on the nodes of the mesh.
  psSmartPointer<lsMesh<NumericType>> calculateFlux() const {

    // Generate disk mesh from domain
    auto mesh = lsSmartPointer<lsMesh<NumericType>>::New();
    lsToDiskMesh<NumericType, D> meshConverter(mesh);
    for (auto dom : *domain->getLevelSets()) {
      meshConverter.insertNextLevelSet(dom);
    }
    meshConverter.apply();

    if (model->getSurfaceModel()->getCoverages() != nullptr) {
      psLogger::getInstance()
          .addWarning(
              "Coverages are not supported for single-pass flux calculation.")
          .print();
      return mesh;
    }

    rayBoundaryCondition rayBoundaryCondition[D];
    rayTrace<NumericType, D> rayTracer;

    // Map the domain boundary to the ray tracing boundaries
    if (ignoreFluxBoundaries) {
      for (unsigned i = 0; i < D; ++i)
        rayBoundaryCondition[i] = rayBoundaryCondition::IGNORE;
    } else {
      for (unsigned i = 0; i < D; ++i)
        rayBoundaryCondition[i] = psUtils::convertBoundaryCondition<D>(
            domain->getGrid().getBoundaryConditions(i));
    }
    rayTracer.setSourceDirection(sourceDirection);
    rayTracer.setNumberOfRaysPerPoint(raysPerPoint);
    rayTracer.setBoundaryConditions(rayBoundaryCondition);
    rayTracer.setUseRandomSeeds(useRandomSeeds_);
    rayTracer.setCalculateFlux(false);
    auto source = model->getSource();
    if (source) {
      rayTracer.setSource(source);
      psLogger::getInstance().addInfo("Using custom source.").print();
    }
    auto primaryDirection = model->getPrimaryDirection();
    if (primaryDirection) {
      psLogger::getInstance()
          .addInfo("Using primary direction: " +
                   psUtils::arrayToString(primaryDirection.value()))
          .print();
      rayTracer.setPrimaryDirection(primaryDirection.value());
    }

    auto points = mesh->getNodes();
    auto normals = *mesh->getCellData().getVectorData("Normals");
    auto materialIds = *mesh->getCellData().getScalarData("MaterialIds");
    rayTracer.setGeometry(points, normals, domain->getGrid().getGridDelta());
    rayTracer.setMaterialIds(materialIds);

    for (auto &particle : model->getParticleTypes()) {
      rayTracer.setParticleType(particle);
      rayTracer.apply();

      // fill up rates vector with rates from this particle type
      auto &localData = rayTracer.getLocalData();
      int numRates = particle->getLocalDataLabels().size();
      for (int i = 0; i < numRates; ++i) {
        auto rate = std::move(localData.getVectorData(i));

        // normalize fluxes
        rayTracer.normalizeFlux(rate);
        if (smoothFlux)
          rayTracer.smoothFlux(rate);
        mesh->getCellData().insertNextScalarData(
            std::move(rate), localData.getVectorDataLabel(i));
      }
    }

    return mesh;
  }

  // Run the process.
  void apply() {
    /* ---------- Process Setup --------- */
    if (!model) {
      psLogger::getInstance()
          .addWarning("No process model passed to psProcess.")
          .print();
      return;
    }
    const auto name = model->getProcessName().value_or("default");

    if (!domain) {
      psLogger::getInstance()
          .addWarning("No domain passed to psProcess.")
          .print();
      return;
    }

    if (model->getGeometricModel()) {
      model->getGeometricModel()->setDomain(domain);
      psLogger::getInstance().addInfo("Applying geometric model...").print();
      model->getGeometricModel()->apply();
      return;
    }

    if (processDuration == 0.) {
      // apply only advection callback
      if (model->getAdvectionCallback()) {
        model->getAdvectionCallback()->setDomain(domain);
        model->getAdvectionCallback()->applyPreAdvect(0);
      } else {
        psLogger::getInstance()
            .addWarning("No advection callback passed to psProcess.")
            .print();
      }
      return;
    }

    if (!model->getSurfaceModel()) {
      psLogger::getInstance()
          .addWarning("No surface model passed to psProcess.")
          .print();
      return;
    }

    if (!model->getVelocityField()) {
      psLogger::getInstance()
          .addWarning("No velocity field passed to psProcess.")
          .print();
      return;
    }

    psUtils::Timer processTimer;
    processTimer.start();

    double remainingTime = processDuration;
    assert(domain->getLevelSets()->size() != 0 && "No level sets in domain.");
    const NumericType gridDelta = domain->getGrid().getGridDelta();

    auto diskMesh = lsSmartPointer<lsMesh<NumericType>>::New();
    auto translator = lsSmartPointer<translatorType>::New();
    lsToDiskMesh<NumericType, D> meshConverter(diskMesh);
    meshConverter.setTranslator(translator);
    if (domain->getMaterialMap() &&
        domain->getMaterialMap()->size() == domain->getLevelSets()->size()) {
      meshConverter.setMaterialMap(domain->getMaterialMap()->getMaterialMap());
    }

    auto transField = psSmartPointer<psTranslationField<NumericType>>::New(
        model->getVelocityField(), domain->getMaterialMap());
    transField->setTranslator(translator);

    lsAdvect<NumericType, D> advectionKernel;
    advectionKernel.setVelocityField(transField);
    advectionKernel.setIntegrationScheme(integrationScheme);
    advectionKernel.setTimeStepRatio(timeStepRatio);

    for (auto dom : *domain->getLevelSets()) {
      meshConverter.insertNextLevelSet(dom);
      advectionKernel.insertNextLevelSet(dom);
    }

    /* --------- Setup for ray tracing ----------- */
    const bool useRayTracing = !model->getParticleTypes().empty();

    rayBoundaryCondition rayBoundaryCondition[D];
    rayTrace<NumericType, D> rayTracer;

    if (useRayTracing) {
      // Map the domain boundary to the ray tracing boundaries
      if (ignoreFluxBoundaries) {
        for (unsigned i = 0; i < D; ++i)
          rayBoundaryCondition[i] = rayBoundaryCondition::IGNORE;
      } else {
        for (unsigned i = 0; i < D; ++i)
          rayBoundaryCondition[i] = psUtils::convertBoundaryCondition<D>(
              domain->getGrid().getBoundaryConditions(i));
      }

      rayTracer.setSourceDirection(sourceDirection);
      rayTracer.setNumberOfRaysPerPoint(raysPerPoint);
      rayTracer.setBoundaryConditions(rayBoundaryCondition);
      rayTracer.setUseRandomSeeds(useRandomSeeds_);
      auto primaryDirection = model->getPrimaryDirection();
      if (primaryDirection) {
        psLogger::getInstance()
            .addInfo("Using primary direction: " +
                     psUtils::arrayToString(primaryDirection.value()))
            .print();
        rayTracer.setPrimaryDirection(primaryDirection.value());
      }
      rayTracer.setCalculateFlux(false);
      auto source = model->getSource();
      if (source) {
        rayTracer.setSource(source);
        psLogger::getInstance().addInfo("Using custom source.").print();
      }

      // initialize particle data logs
      particleDataLogs.resize(model->getParticleTypes().size());
      for (std::size_t i = 0; i < model->getParticleTypes().size(); i++) {
        int logSize = model->getParticleLogSize(i);
        if (logSize > 0) {
          particleDataLogs[i].data.resize(1);
          particleDataLogs[i].data[0].resize(logSize);
        }
      }
    }

    // Determine whether advection callback is used
    const bool useAdvectionCallback = model->getAdvectionCallback() != nullptr;
    if (useAdvectionCallback) {
      model->getAdvectionCallback()->setDomain(domain);
    }

    // Determine whether there are process parameters used in ray tracing
    model->getSurfaceModel()->initializeProcessParameters();
    const bool useProcessParams =
        model->getSurfaceModel()->getProcessParameters() != nullptr;

    if (useProcessParams)
      psLogger::getInstance().addInfo("Using process parameters.").print();
    if (useAdvectionCallback)
      psLogger::getInstance().addInfo("Using advection callback.").print();

    bool useCoverages = false;

    // Initialize coverages
    meshConverter.apply();
    auto numPoints = diskMesh->getNodes().size();
    if (!coveragesInitialized_)
      model->getSurfaceModel()->initializeCoverages(numPoints);
    if (model->getSurfaceModel()->getCoverages() != nullptr) {
      psUtils::Timer timer;
      useCoverages = true;
      psLogger::getInstance().addInfo("Using coverages.").print();
      if (!coveragesInitialized_) {
        timer.start();
        psLogger::getInstance().addInfo("Initializing coverages ... ").print();
        auto points = diskMesh->getNodes();
        auto normals = *diskMesh->getCellData().getVectorData("Normals");
        auto materialIds =
            *diskMesh->getCellData().getScalarData("MaterialIds");
        rayTracer.setGeometry(points, normals, gridDelta);
        rayTracer.setMaterialIds(materialIds);

        for (size_t iterations = 0; iterations < maxIterations; iterations++) {
          // We need additional signal handling when running the C++ code from
          // the
          // Python bindings to allow interrupts in the Python scripts
#ifdef VIENNAPS_PYTHON_BUILD
          if (PyErr_CheckSignals() != 0)
            throw pybind11::error_already_set();
#endif
          // move coverages to the ray tracer
          rayTracingData<NumericType> rayTraceCoverages =
              movePointDataToRayData(model->getSurfaceModel()->getCoverages());
          if (useProcessParams) {
            // store scalars in addition to coverages
            auto processParams =
                model->getSurfaceModel()->getProcessParameters();
            NumericType numParams = processParams->getScalarData().size();
            rayTraceCoverages.setNumberOfScalarData(numParams);
            for (size_t i = 0; i < numParams; ++i) {
              rayTraceCoverages.setScalarData(
                  i, processParams->getScalarData(i),
                  processParams->getScalarDataLabel(i));
            }
          }
          rayTracer.setGlobalData(rayTraceCoverages);

          auto rates = psSmartPointer<psPointData<NumericType>>::New();

          std::size_t particleIdx = 0;
          for (auto &particle : model->getParticleTypes()) {
            int dataLogSize = model->getParticleLogSize(particleIdx);
            if (dataLogSize > 0) {
              rayTracer.getDataLog().data.resize(1);
              rayTracer.getDataLog().data[0].resize(dataLogSize, 0.);
            }
            rayTracer.setParticleType(particle);
            rayTracer.apply();

            // fill up rates vector with rates from this particle type
            auto &localData = rayTracer.getLocalData();
            int numRates = particle->getLocalDataLabels().size();
            for (int i = 0; i < numRates; ++i) {
              auto rate = std::move(localData.getVectorData(i));

              // normalize fluxes
              rayTracer.normalizeFlux(rate);
              if (smoothFlux)
                rayTracer.smoothFlux(rate);
              rates->insertNextScalarData(std::move(rate),
                                          localData.getVectorDataLabel(i));
            }

            if (dataLogSize > 0) {
              particleDataLogs[particleIdx].merge(rayTracer.getDataLog());
            }
            ++particleIdx;
          }

          // move coverages back in the model
          moveRayDataToPointData(model->getSurfaceModel()->getCoverages(),
                                 rayTraceCoverages);
          model->getSurfaceModel()->updateCoverages(rates, materialIds);

          if (psLogger::getLogLevel() >= 3) {
            auto coverages = model->getSurfaceModel()->getCoverages();
            for (size_t idx = 0; idx < coverages->getScalarDataSize(); idx++) {
              auto label = coverages->getScalarDataLabel(idx);
              diskMesh->getCellData().insertNextScalarData(
                  *coverages->getScalarData(idx), label);
            }
            for (size_t idx = 0; idx < rates->getScalarDataSize(); idx++) {
              auto label = rates->getScalarDataLabel(idx);
              diskMesh->getCellData().insertNextScalarData(
                  *rates->getScalarData(idx), label);
            }
            printDiskMesh(diskMesh, name + "_covIinit_" +
                                        std::to_string(iterations) + ".vtp");
            psLogger::getInstance()
                .addInfo("Iteration: " + std::to_string(iterations))
                .print();
          }
        }
        coveragesInitialized_ = true;

        timer.finish();
        psLogger::getInstance()
            .addTiming("Coverage initialization", timer)
            .print();
      }
    } // end coverage initialization

    double previousTimeStep = 0.;
    size_t counter = 0;
    psUtils::Timer rtTimer;
    psUtils::Timer callbackTimer;
    psUtils::Timer advTimer;
    while (remainingTime > 0.) {
      // We need additional signal handling when running the C++ code from the
      // Python bindings to allow interrupts in the Python scripts
#ifdef VIENNAPS_PYTHON_BUILD
      if (PyErr_CheckSignals() != 0)
        throw pybind11::error_already_set();
#endif

      auto rates = psSmartPointer<psPointData<NumericType>>::New();
      meshConverter.apply();
      auto materialIds = *diskMesh->getCellData().getScalarData("MaterialIds");
      auto points = diskMesh->getNodes();

      // rate calculation by top-down ray tracing
      if (useRayTracing) {
        rtTimer.start();
        auto normals = *diskMesh->getCellData().getVectorData("Normals");
        rayTracer.setGeometry(points, normals, gridDelta);
        rayTracer.setMaterialIds(materialIds);

        // move coverages to ray tracer
        rayTracingData<NumericType> rayTraceCoverages;
        if (useCoverages) {
          rayTraceCoverages =
              movePointDataToRayData(model->getSurfaceModel()->getCoverages());
          if (useProcessParams) {
            // store scalars in addition to coverages
            auto processParams =
                model->getSurfaceModel()->getProcessParameters();
            NumericType numParams = processParams->getScalarData().size();
            rayTraceCoverages.setNumberOfScalarData(numParams);
            for (size_t i = 0; i < numParams; ++i) {
              rayTraceCoverages.setScalarData(
                  i, processParams->getScalarData(i),
                  processParams->getScalarDataLabel(i));
            }
          }
          rayTracer.setGlobalData(rayTraceCoverages);
        }

        std::size_t particleIdx = 0;
        for (auto &particle : model->getParticleTypes()) {
          int dataLogSize = model->getParticleLogSize(particleIdx);
          if (dataLogSize > 0) {
            rayTracer.getDataLog().data.resize(1);
            rayTracer.getDataLog().data[0].resize(dataLogSize, 0.);
          }
          rayTracer.setParticleType(particle);
          rayTracer.apply();

          // fill up rates vector with rates from this particle type
          auto numRates = particle->getLocalDataLabels().size();
          auto &localData = rayTracer.getLocalData();
          for (int i = 0; i < numRates; ++i) {
            auto rate = std::move(localData.getVectorData(i));

            // normalize rates
            rayTracer.normalizeFlux(rate);
            if (smoothFlux)
              rayTracer.smoothFlux(rate);
            rates->insertNextScalarData(std::move(rate),
                                        localData.getVectorDataLabel(i));
          }

          if (dataLogSize > 0) {
            particleDataLogs[particleIdx].merge(rayTracer.getDataLog());
          }
          ++particleIdx;
        }

        // move coverages back to model
        if (useCoverages)
          moveRayDataToPointData(model->getSurfaceModel()->getCoverages(),
                                 rayTraceCoverages);
        rtTimer.finish();
        psLogger::getInstance()
            .addTiming("Top-down flux calculation", rtTimer)
            .print();
      }

      // get velocities from rates
      auto velocities = model->getSurfaceModel()->calculateVelocities(
          rates, points, materialIds);
      model->getVelocityField()->setVelocities(velocities);
      if (model->getVelocityField()->getTranslationFieldOptions() == 2)
        transField->buildKdTree(points);

      // print debug output
      if (psLogger::getLogLevel() >= 4) {
        if (printTime >= 0. &&
            ((processDuration - remainingTime) - printTime * counter) > 0.) {
          if (velocities)
            diskMesh->getCellData().insertNextScalarData(*velocities,
                                                         "velocities");
          if (useCoverages) {
            auto coverages = model->getSurfaceModel()->getCoverages();
            for (size_t idx = 0; idx < coverages->getScalarDataSize(); idx++) {
              auto label = coverages->getScalarDataLabel(idx);
              diskMesh->getCellData().insertNextScalarData(
                  *coverages->getScalarData(idx), label);
            }
          }
          for (size_t idx = 0; idx < rates->getScalarDataSize(); idx++) {
            auto label = rates->getScalarDataLabel(idx);
            diskMesh->getCellData().insertNextScalarData(
                *rates->getScalarData(idx), label);
          }
          printDiskMesh(diskMesh,
                        name + "_" + std::to_string(counter) + ".vtp");
          if (domain->getCellSet()) {
            domain->getCellSet()->writeVTU(name + "_cellSet_" +
                                           std::to_string(counter) + ".vtu");
          }
          counter++;
        }
      }

      // apply advection callback
      if (useAdvectionCallback) {
        callbackTimer.start();
        bool continueProcess = model->getAdvectionCallback()->applyPreAdvect(
            processDuration - remainingTime);
        callbackTimer.finish();
        psLogger::getInstance()
            .addTiming("Advection callback pre-advect", callbackTimer)
            .print();

        if (!continueProcess) {
          psLogger::getInstance()
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

      // move coverages to LS, so they get are moved with the advection step
      if (useCoverages)
        moveCoveragesToTopLS(translator,
                             model->getSurfaceModel()->getCoverages());
      advTimer.start();
      advectionKernel.apply();
      advTimer.finish();
      psLogger::getInstance().addTiming("Surface advection", advTimer).print();

      // update the translator to retrieve the correct coverages from the LS
      meshConverter.apply();
      if (useCoverages)
        updateCoveragesFromAdvectedSurface(
            translator, model->getSurfaceModel()->getCoverages());

      // apply advection callback
      if (useAdvectionCallback) {
        callbackTimer.start();
        bool continueProcess = model->getAdvectionCallback()->applyPostAdvect(
            advectionKernel.getAdvectedTime());
        callbackTimer.finish();
        psLogger::getInstance()
            .addTiming("Advection callback post-advect", callbackTimer)
            .print();
        if (!continueProcess) {
          psLogger::getInstance()
              .addInfo("Process stopped early by AdvectionCallback during "
                       "`postAdvect`.")
              .print();
          break;
        }
      }

      previousTimeStep = advectionKernel.getAdvectedTime();
      remainingTime -= previousTimeStep;

      if (psLogger::getLogLevel() >= 2) {
        std::stringstream stream;
        stream << std::fixed << std::setprecision(4)
               << "Process time: " << processDuration - remainingTime << " / "
               << processDuration;
        psLogger::getInstance().addInfo(stream.str()).print();
      }
    }

    processTime = processDuration - remainingTime;
    processTimer.finish();

    psLogger::getInstance()
        .addTiming("\nProcess " + name, processTimer)
        .addTiming("Surface advection total time",
                   advTimer.totalDuration * 1e-9,
                   processTimer.totalDuration * 1e-9)
        .print();
    if (useRayTracing) {
      psLogger::getInstance()
          .addTiming("Top-down flux calculation total time",
                     rtTimer.totalDuration * 1e-9,
                     processTimer.totalDuration * 1e-9)
          .print();
    }
    if (useAdvectionCallback) {
      psLogger::getInstance()
          .addTiming("Advection callback total time",
                     callbackTimer.totalDuration * 1e-9,
                     processTimer.totalDuration * 1e-9)
          .print();
    }
  }

  void writeParticleDataLogs(std::string fileName) {
    std::ofstream file(fileName.c_str());

    for (std::size_t i = 0; i < particleDataLogs.size(); i++) {
      if (!particleDataLogs[i].data.empty()) {
        file << "particle" << i << "_data ";
        for (std::size_t j = 0; j < particleDataLogs[i].data[0].size(); j++) {
          file << particleDataLogs[i].data[0][j] << " ";
        }
        file << "\n";
      }
    }

    file.close();
  }

private:
  void printDiskMesh(lsSmartPointer<lsMesh<NumericType>> mesh,
                     std::string name) const {
    psVTKWriter<NumericType>(mesh, std::move(name)).apply();
  }

  rayTracingData<NumericType> movePointDataToRayData(
      psSmartPointer<psPointData<NumericType>> pointData) const {
    rayTracingData<NumericType> rayData;
    const auto numData = pointData->getScalarDataSize();
    rayData.setNumberOfVectorData(numData);
    for (size_t i = 0; i < numData; ++i) {
      auto label = pointData->getScalarDataLabel(i);
      rayData.setVectorData(i, std::move(*pointData->getScalarData(label)),
                            label);
    }

    return std::move(rayData);
  }

  void
  moveRayDataToPointData(psSmartPointer<psPointData<NumericType>> pointData,
                         rayTracingData<NumericType> &rayData) const {
    pointData->clear();
    const auto numData = rayData.getVectorData().size();
    for (size_t i = 0; i < numData; ++i)
      pointData->insertNextScalarData(std::move(rayData.getVectorData(i)),
                                      rayData.getVectorDataLabel(i));
  }

  void
  moveCoveragesToTopLS(lsSmartPointer<translatorType> translator,
                       psSmartPointer<psPointData<NumericType>> coverages) {
    auto topLS = domain->getLevelSets()->back();
    for (size_t i = 0; i < coverages->getScalarDataSize(); i++) {
      auto covName = coverages->getScalarDataLabel(i);
      std::vector<NumericType> levelSetData(topLS->getNumberOfPoints(), 0);
      auto cov = coverages->getScalarData(covName);
      for (const auto iter : *translator.get()) {
        levelSetData[iter.first] = cov->at(iter.second);
      }
      if (auto data = topLS->getPointData().getScalarData(covName);
          data != nullptr) {
        *data = std::move(levelSetData);
      } else {
        topLS->getPointData().insertNextScalarData(std::move(levelSetData),
                                                   covName);
      }
    }
  }

  void updateCoveragesFromAdvectedSurface(
      lsSmartPointer<translatorType> translator,
      psSmartPointer<psPointData<NumericType>> coverages) const {
    auto topLS = domain->getLevelSets()->back();
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

  psDomainType domain;
  psSmartPointer<psProcessModel<NumericType, D>> model;
  NumericType processDuration;
  rayTraceDirection sourceDirection =
      D == 3 ? rayTraceDirection::POS_Z : rayTraceDirection::POS_Y;
  lsIntegrationSchemeEnum integrationScheme =
      lsIntegrationSchemeEnum::ENGQUIST_OSHER_1ST_ORDER;
  unsigned raysPerPoint = 1000;
  std::vector<rayDataLog<NumericType>> particleDataLogs;
  bool useRandomSeeds_ = true;
  bool smoothFlux = true;
  bool ignoreFluxBoundaries = false;
  unsigned maxIterations = 20;
  bool coveragesInitialized_ = false;
  NumericType printTime = 0.;
  NumericType processTime = 0.;
  NumericType timeStepRatio = 0.4999;
};
