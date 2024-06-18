#pragma once

#include "psProcessModel.hpp"
#include "psTranslationField.hpp"
#include "psUtils.hpp"

#include <lsAdvect.hpp>
#include <lsDomain.hpp>
#include <lsMesh.hpp>
#include <lsToDiskMesh.hpp>
#include <lsVTKWriter.hpp>

#include <rayParticle.hpp>
#include <rayTrace.hpp>

namespace viennaps {

using namespace viennacore;

/// This class server as the main process tool, applying a user- or pre-defined
/// process model to a domain. Depending on the user inputs surface advection, a
/// single callback function or a geometric advection is applied.
template <typename NumericType, int D> class Process {
  using translatorType = std::unordered_map<unsigned long, unsigned long>;
  using psDomainType = SmartPointer<Domain<NumericType, D>>;

public:
  Process() {}
  Process(psDomainType passedDomain) : domain(passedDomain) {}
  // Constructor for a process with a pre-configured process model.
  Process(psDomainType passedDomain,
          SmartPointer<ProcessModel<NumericType, D>> passedProcessModel,
          const NumericType passedDuration = 0.)
      : domain(passedDomain), model(passedProcessModel),
        processDuration(passedDuration) {}

  // Set the process model. This can be either a pre-configured process model or
  // a custom process model. A custom process model must interface the
  // psProcessModel class.
  void setProcessModel(
      SmartPointer<ProcessModel<NumericType, D>> passedProcessModel) {
    model = passedProcessModel;
  }

  // Set the process domain.
  void setDomain(psDomainType passedDomain) { domain = passedDomain; }

  // Set the source direction, where the rays should be traced from.
  void setSourceDirection(viennaray::TraceDirection passedDirection) {
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
  // Possible integration schemes are specified in
  // viennals::IntegrationSchemeEnum.
  void setIntegrationScheme(
      viennals::IntegrationSchemeEnum passedIntegrationScheme) {
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
  SmartPointer<viennals::Mesh<NumericType>> calculateFlux() const {

    // Generate disk mesh from domain
    auto mesh = SmartPointer<viennals::Mesh<NumericType>>::New();
    viennals::ToDiskMesh<NumericType, D> meshConverter(mesh);
    for (auto dom : domain->getLevelSets()) {
      meshConverter.insertNextLevelSet(dom);
    }
    meshConverter.apply();

    if (model->getSurfaceModel()->getCoverages() != nullptr) {
      Logger::getInstance()
          .addWarning(
              "Coverages are not supported for single-pass flux calculation.")
          .print();
      return mesh;
    }

    viennaray::BoundaryCondition rayBoundaryCondition[D];
    viennaray::Trace<NumericType, D> rayTracer;

    // Map the domain boundary to the ray tracing boundaries
    if (ignoreFluxBoundaries) {
      for (unsigned i = 0; i < D; ++i)
        rayBoundaryCondition[i] = viennaray::BoundaryCondition::IGNORE;
    } else {
      for (unsigned i = 0; i < D; ++i)
        rayBoundaryCondition[i] = utils::convertBoundaryCondition<D>(
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
      Logger::getInstance().addInfo("Using custom source.").print();
    }
    auto primaryDirection = model->getPrimaryDirection();
    if (primaryDirection) {
      Logger::getInstance()
          .addInfo("Using primary direction: " +
                   utils::arrayToString(primaryDirection.value()))
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
      Logger::getInstance()
          .addWarning("No process model passed to psProcess.")
          .print();
      return;
    }
    const auto name = model->getProcessName().value_or("default");

    if (!domain) {
      Logger::getInstance()
          .addWarning("No domain passed to psProcess.")
          .print();
      return;
    }

    if (model->getGeometricModel()) {
      model->getGeometricModel()->setDomain(domain);
      Logger::getInstance().addInfo("Applying geometric model...").print();
      model->getGeometricModel()->apply();
      return;
    }

    if (processDuration == 0.) {
      // apply only advection callback
      if (model->getAdvectionCallback()) {
        model->getAdvectionCallback()->setDomain(domain);
        model->getAdvectionCallback()->applyPreAdvect(0);
      } else {
        Logger::getInstance()
            .addWarning("No advection callback passed to psProcess.")
            .print();
      }
      return;
    }

    if (!model->getSurfaceModel()) {
      Logger::getInstance()
          .addWarning("No surface model passed to psProcess.")
          .print();
      return;
    }

    if (!model->getVelocityField()) {
      Logger::getInstance()
          .addWarning("No velocity field passed to psProcess.")
          .print();
      return;
    }

    Timer processTimer;
    processTimer.start();

    double remainingTime = processDuration;
    assert(domain->getLevelSets().size() != 0 && "No level sets in domain.");
    const NumericType gridDelta = domain->getGrid().getGridDelta();

    auto diskMesh = SmartPointer<viennals::Mesh<NumericType>>::New();
    auto translator = SmartPointer<translatorType>::New();
    viennals::ToDiskMesh<NumericType, D> meshConverter(diskMesh);
    meshConverter.setTranslator(translator);
    if (domain->getMaterialMap() &&
        domain->getMaterialMap()->size() == domain->getLevelSets().size()) {
      meshConverter.setMaterialMap(domain->getMaterialMap()->getMaterialMap());
    }

    auto transField = SmartPointer<TranslationField<NumericType>>::New(
        model->getVelocityField(), domain->getMaterialMap());
    transField->setTranslator(translator);

    viennals::Advect<NumericType, D> advectionKernel;
    advectionKernel.setVelocityField(transField);
    advectionKernel.setIntegrationScheme(integrationScheme);
    advectionKernel.setTimeStepRatio(timeStepRatio);

    for (auto dom : domain->getLevelSets()) {
      meshConverter.insertNextLevelSet(dom);
      advectionKernel.insertNextLevelSet(dom);
    }

    /* --------- Setup for ray tracing ----------- */
    const bool useRayTracing = !model->getParticleTypes().empty();

    viennaray::BoundaryCondition rayBoundaryCondition[D];
    viennaray::Trace<NumericType, D> rayTracer;

    if (useRayTracing) {
      // Map the domain boundary to the ray tracing boundaries
      if (ignoreFluxBoundaries) {
        for (unsigned i = 0; i < D; ++i)
          rayBoundaryCondition[i] = viennaray::BoundaryCondition::IGNORE;
      } else {
        for (unsigned i = 0; i < D; ++i)
          rayBoundaryCondition[i] = utils::convertBoundaryCondition<D>(
              domain->getGrid().getBoundaryConditions(i));
      }

      rayTracer.setSourceDirection(sourceDirection);
      rayTracer.setNumberOfRaysPerPoint(raysPerPoint);
      rayTracer.setBoundaryConditions(rayBoundaryCondition);
      rayTracer.setUseRandomSeeds(useRandomSeeds_);
      auto primaryDirection = model->getPrimaryDirection();
      if (primaryDirection) {
        Logger::getInstance()
            .addInfo("Using primary direction: " +
                     utils::arrayToString(primaryDirection.value()))
            .print();
        rayTracer.setPrimaryDirection(primaryDirection.value());
      }
      rayTracer.setCalculateFlux(false);
      auto source = model->getSource();
      if (source) {
        rayTracer.setSource(source);
        Logger::getInstance().addInfo("Using custom source.").print();
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
      Logger::getInstance().addInfo("Using process parameters.").print();
    if (useAdvectionCallback)
      Logger::getInstance().addInfo("Using advection callback.").print();

    bool useCoverages = false;

    // Initialize coverages
    meshConverter.apply();
    auto numPoints = diskMesh->getNodes().size();
    if (!coveragesInitialized_)
      model->getSurfaceModel()->initializeCoverages(numPoints);
    if (model->getSurfaceModel()->getCoverages() != nullptr) {
      Timer timer;
      useCoverages = true;
      Logger::getInstance().addInfo("Using coverages.").print();
      if (!coveragesInitialized_) {
        timer.start();
        Logger::getInstance().addInfo("Initializing coverages ... ").print();
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
          viennaray::TracingData<NumericType> rayTraceCoverages =
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

          auto rates = SmartPointer<viennals::PointData<NumericType>>::New();

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

          if (Logger::getLogLevel() >= 3) {
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
            Logger::getInstance()
                .addInfo("Iteration: " + std::to_string(iterations))
                .print();
          }
        }
        coveragesInitialized_ = true;

        timer.finish();
        Logger::getInstance()
            .addTiming("Coverage initialization", timer)
            .print();
      }
    } // end coverage initialization

    double previousTimeStep = 0.;
    size_t counter = 0;
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

      auto rates = SmartPointer<viennals::PointData<NumericType>>::New();
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
        viennaray::TracingData<NumericType> rayTraceCoverages;
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
        Logger::getInstance()
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
      if (Logger::getLogLevel() >= 4) {
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

      // move coverages to LS, so they get are moved with the advection step
      if (useCoverages)
        moveCoveragesToTopLS(translator,
                             model->getSurfaceModel()->getCoverages());
      advTimer.start();
      advectionKernel.apply();
      advTimer.finish();
      Logger::getInstance().addTiming("Surface advection", advTimer).print();

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
      remainingTime -= previousTimeStep;

      if (Logger::getLogLevel() >= 2) {
        std::stringstream stream;
        stream << std::fixed << std::setprecision(4)
               << "Process time: " << processDuration - remainingTime << " / "
               << processDuration;
        Logger::getInstance().addInfo(stream.str()).print();
      }
    }

    processTime = processDuration - remainingTime;
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
  void printDiskMesh(SmartPointer<viennals::Mesh<NumericType>> mesh,
                     std::string name) const {
    viennals::VTKWriter<NumericType>(mesh, std::move(name)).apply();
  }

  viennaray::TracingData<NumericType> movePointDataToRayData(
      SmartPointer<viennals::PointData<NumericType>> pointData) const {
    viennaray::TracingData<NumericType> rayData;
    const auto numData = pointData->getScalarDataSize();
    rayData.setNumberOfVectorData(numData);
    for (size_t i = 0; i < numData; ++i) {
      auto label = pointData->getScalarDataLabel(i);
      rayData.setVectorData(i, std::move(*pointData->getScalarData(label)),
                            label);
    }

    return std::move(rayData);
  }

  void moveRayDataToPointData(
      SmartPointer<viennals::PointData<NumericType>> pointData,
      viennaray::TracingData<NumericType> &rayData) const {
    pointData->clear();
    const auto numData = rayData.getVectorData().size();
    for (size_t i = 0; i < numData; ++i)
      pointData->insertNextScalarData(std::move(rayData.getVectorData(i)),
                                      rayData.getVectorDataLabel(i));
  }

  void moveCoveragesToTopLS(
      SmartPointer<translatorType> translator,
      SmartPointer<viennals::PointData<NumericType>> coverages) {
    auto topLS = domain->getLevelSets().back();
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
      SmartPointer<translatorType> translator,
      SmartPointer<viennals::PointData<NumericType>> coverages) const {
    auto topLS = domain->getLevelSets().back();
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
  SmartPointer<ProcessModel<NumericType, D>> model;
  NumericType processDuration;
  viennaray::TraceDirection sourceDirection =
      D == 3 ? viennaray::TraceDirection::POS_Z
             : viennaray::TraceDirection::POS_Y;
  viennals::IntegrationSchemeEnum integrationScheme =
      viennals::IntegrationSchemeEnum::ENGQUIST_OSHER_1ST_ORDER;
  unsigned raysPerPoint = 1000;
  std::vector<viennaray::DataLog<NumericType>> particleDataLogs;
  bool useRandomSeeds_ = true;
  bool smoothFlux = true;
  bool ignoreFluxBoundaries = false;
  unsigned maxIterations = 20;
  bool coveragesInitialized_ = false;
  NumericType printTime = 0.;
  NumericType processTime = 0.;
  NumericType timeStepRatio = 0.4999;
};

} // namespace viennaps
