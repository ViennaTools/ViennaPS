#pragma once

#include <cassert>
#include <cstring>

#include <context.hpp>

#include <lsAdvect.hpp>
#include <lsDomain.hpp>
#include <lsMesh.hpp>
#include <lsToDiskMesh.hpp>

#include <pscuProcessModel.hpp>
#include <pscuSurfaceModel.hpp>

#include <psAdvectionCallback.hpp>
#include <psDomain.hpp>
#include <psLogger.hpp>
#include <psSmartPointer.hpp>
#include <psTranslationField.hpp>
#include <psVelocityField.hpp>

#include <curtIndexMap.hpp>
#include <curtParticle.hpp>
#include <curtSmoothing.hpp>
#include <curtTracer.hpp>

#include <culsToSurfaceMesh.hpp>

template <typename NumericType, int D> class pscuProcess {
  using translatorType = std::unordered_map<unsigned long, unsigned long>;
  using psDomainType = psSmartPointer<psDomain<NumericType, D>>;

public:
  pscuProcess(pscuContext passedContext) : context(passedContext) {}

  template <typename ProcessModelType>
  void setProcessModel(psSmartPointer<ProcessModelType> passedProcessModel) {
    model = std::dynamic_pointer_cast<pscuProcessModel<NumericType>>(
        passedProcessModel);
  }

  void setDomain(psSmartPointer<psDomain<NumericType, D>> passedDomain) {
    domain = passedDomain;
  }

  void setProcessDuration(double passedDuration) {
    processDuration = passedDuration;
  }

  void setNumberOfRaysPerPoint(long numRays) { raysPerPoint = numRays; }

  void setMaxCoverageInitIterations(size_t maxIt) { maxIterations = maxIt; }

  void setPeriodicBoundary(const int passedPeriodic) {
    periodicBoundary = static_cast<bool>(passedPeriodic);
  }

  void apply() {
    /* ---------- Process Setup --------- */
    if (!model) {
      psLogger::getInstance()
          .addWarning("No process model passed to psProcess.")
          .print();
      return;
    }
    const auto name = model->getProcessName();

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

    psUtils::Timer processTimer;
    processTimer.start();

    double remainingTime = processDuration;
    assert(domain->getLevelSets()->size() != 0 && "No level sets in domain.");
    const NumericType gridDelta =
        domain->getLevelSets()->back()->getGrid().getGridDelta();
    utCudaBuffer d_rates;

    auto diskMesh = psSmartPointer<lsMesh<NumericType>>::New();
    auto translator = psSmartPointer<translatorType>::New();
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

    for (auto dom : *domain->getLevelSets()) {
      meshConverter.insertNextLevelSet(dom);
      advectionKernel.insertNextLevelSet(dom);
    }

    /* --------- Setup for ray tracing ----------- */
    const bool useRayTracing = model->getParticleTypes() != nullptr;
    auto kdTree = psSmartPointer<
        psKDTree<NumericType, std::array<NumericType, 3>>>::New();

    if (useRayTracing && !rayTracerInitialized) {
      if (!model->getPtxCode()) {
        psLogger::getInstance()
            .addWarning("No pipeline in process model. Aborting.")
            .print();
        return;
      }
      rayTrace.setKdTree(kdTree);
      rayTrace.setPipeline(model->getPtxCode());
      rayTrace.setLevelSet(domain->getLevelSets()->back());
      rayTrace.setNumberOfRaysPerPoint(raysPerPoint);
      rayTrace.setUseRandomSeed(useRandomSeeds);
      rayTrace.setPeriodicBoundary(periodicBoundary);
      for (auto &particle : *model->getParticleTypes()) {
        rayTrace.insertNextParticle(particle);
      }
      rayTrace.prepareParticlePrograms();
      model->getSurfaceModel()->setIndexMap(
          curtIndexMap<NumericType>::getPointDataIndexMap(
              rayTrace.getParticles()));
    }

    // Determine whether advection callback is used
    const bool useAdvectionCallback = model->getAdvectionCallback() != nullptr;
    if (useAdvectionCallback) {
      model->getAdvectionCallback()->setDomain(domain);
    }

    // Determine whether there are process parameters used in ray tracing
    model->getSurfaceModel()->initializeProcessParameters();
    const bool useProcessParams =
        model->getSurfaceModel()->getProcessParameters().d_ptr != nullptr;

    if (useProcessParams)
      psLogger::getInstance().addInfo("Using process parameters.").print();
    if (useAdvectionCallback)
      psLogger::getInstance().addInfo("Using advection callback.").print();

    bool useCoverages = false;
    size_t numCov = 0;
    // Initialize coverages
    meshConverter.apply();
    auto numPoints = diskMesh->getNodes().size();
    if (!coveragesInitialized)
      model->getSurfaceModel()->initializeCoverages(numPoints);
    if (model->getSurfaceModel()->getCoverages().d_ptr != nullptr) {
      useCoverages = true;
      numCov = model->getSurfaceModel()->getCoverageIndexMap().size();
      rayTrace.setUseCellData(numCov);

      psLogger::getInstance().addInfo("Using coverages.").print();

      if (!coveragesInitialized) {
        psUtils::Timer timer;
        psLogger::getInstance().addInfo("Initializing coverages ... ").print();

        timer.start();
        for (size_t iterations = 1; iterations <= maxIterations; iterations++) {
          // get coverages in ray tracer
          rayTrace.translateFromPointData(
              diskMesh, model->getSurfaceModel()->getCoverages(), numCov);

          rayTrace.apply();
          // get rates
          rayTrace.translateToPointData(diskMesh, d_rates);
          // calculate coverages
          model->getSurfaceModel()->updateCoverages(d_rates,
                                                    diskMesh->nodes.size());

          if (psLogger::getLogLevel() >= 3) {
            rayTrace.downloadResultsToPointData(
                diskMesh->getCellData(), d_rates, diskMesh->nodes.size());
            downloadCoverages(diskMesh->getCellData(),
                              model->getSurfaceModel()->getCoverages(),
                              diskMesh->nodes.size());
            printDiskMesh(diskMesh, name + "_covIinit_" +
                                        std::to_string(iterations) + ".vtp");
            psLogger::getInstance()
                .addInfo("Iteration: " + std::to_string(iterations))
                .print();
          }
        }
        coveragesInitialized = true;

        timer.finish();
        psLogger::getInstance()
            .addTiming("Coverage initialization", timer)
            .print();
      }
    }

    double previousTimeStep = 0.;
    size_t counter = 0;
    size_t printCounter = 0;
    psUtils::Timer rtTimer;
    psUtils::Timer callbackTimer;
    psUtils::Timer advTimer;
    while (remainingTime > 0.) {
      psLogger::getInstance()
          .addInfo("Remaining time: " + std::to_string(remainingTime))
          .print();

      meshConverter.apply();
      auto materialIds = *diskMesh->getCellData().getScalarData("MaterialIds");

      if (useRayTracing) {
        rtTimer.start();

        // get coverages in ray tracer
        if (useCoverages)
          rayTrace.translateFromPointData(
              diskMesh, model->getSurfaceModel()->getCoverages(), numCov);
        rayTrace.apply();

        // TODO "smooth" results on elements
        // TODO: keep results on elements and generate kd tree instead

        // get results as point data
        rayTrace.translateToPointData(diskMesh, d_rates);

        rtTimer.finish();
        psLogger::getInstance()
            .addTiming("Top-down flux calculation", rtTimer)
            .print();
      }

      auto velocities = model->getSurfaceModel()->calculateVelocities(
          d_rates, diskMesh->nodes, materialIds);
      curtSmoothing<NumericType, D>(diskMesh, velocities, gridDelta).apply();
      model->getVelocityField()->setVelocities(velocities);
      if (model->getVelocityField()->getTranslationFieldOptions() == 2)
        transField->buildKdTree(diskMesh->nodes);

      if (psLogger::getLogLevel() >= 4) {
        if (printTime >= 0. &&
            ((processDuration - remainingTime) - printTime * counter) > -1.) {

          rayTrace.downloadResultsToPointData(diskMesh->getCellData(), d_rates,
                                              diskMesh->nodes.size());
          if (useCoverages) {
            downloadCoverages(diskMesh->getCellData(),
                              model->getSurfaceModel()->getCoverages(),
                              diskMesh->nodes.size());
          }

          if (velocities)
            diskMesh->getCellData().insertNextScalarData(*velocities,
                                                         "velocities");

          printDiskMesh(diskMesh,
                        name + "_" + std::to_string(counter) + ".vtp");
          if (domain->getUseCellSet()) {
            domain->getCellSet()->writeVTU(name + "_cellSet_" +
                                           std::to_string(counter) + ".vtu");
          }

          auto mesh = psSmartPointer<lsMesh<NumericType>>::New();
          culsToSurfaceMesh<NumericType>(domain->getLevelSets()->back(), mesh)
              .apply();

          std::vector<NumericType> cellflux(mesh->triangles.size());
          rayTrace.getFlux(cellflux.data(), 0, 0);

          mesh->cellData.insertNextScalarData(cellflux, "flux");
          lsVTKWriter<NumericType>(mesh, name + "_step_" +
                                             std::to_string(counter) + ".vtp")
              .apply();

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
      psPointData<NumericType> coverages;
      if (useCoverages) {
        downloadCoverages(coverages, model->getSurfaceModel()->getCoverages(),
                          diskMesh->nodes.size());
        moveCoveragesToTopLS(domain, translator, coverages);
      }

      advTimer.start();
      advectionKernel.apply();
      advTimer.finish();
      rayTrace.updateSurface();

      psLogger::getInstance().addTiming("Surface advection", advTimer).print();

      // update the translator to retrieve the correct coverages from the LS
      meshConverter.apply();
      if (useCoverages) {
        updateCoveragesFromAdvectedSurface(domain, translator, coverages);
        uploadCoverages(coverages);
      }

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
    } // main process loop
    rayTrace.invalidateGeometry();

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

private:
  void printDiskMesh(psSmartPointer<lsMesh<NumericType>> mesh,
                     std::string name) {
    lsVTKWriter<NumericType>(mesh, name).apply();
  }

  void downloadCoverages(psPointData<NumericType> &coverages,
                         utCudaBuffer &d_coverages, size_t covSize) {
    auto covIndexMap = model->getSurfaceModel()->getCoverageIndexMap();
    const auto numCov = covIndexMap.size();
    NumericType *temp = new NumericType[covSize * numCov];
    d_coverages.download(temp, covSize * numCov);

    for (auto &i : covIndexMap) {
      auto cov = coverages.getScalarData(i.first);
      if (cov == nullptr) {
        std::vector<NumericType> covInit(covSize);
        coverages.insertNextScalarData(std::move(covInit), i.first);
        cov = coverages.getScalarData(i.first);
      }
      if (cov->size() != covSize)
        cov->resize(covSize);
      std::memcpy(cov->data(), temp + i.second * covSize,
                  covSize * sizeof(NumericType));
    }
  }

  void uploadCoverages(psPointData<NumericType> &coverages) {
    std::vector<NumericType> flattenedCoverages;
    assert(coverages.getScalarData(0) != nullptr);
    const auto covSize = coverages.getScalarData(0)->size();
    const auto numCoverages = coverages.getScalarDataSize();
    flattenedCoverages.resize(covSize * numCoverages);

    auto covIndexMap = model->getSurfaceModel()->getCoverageIndexMap();
    for (auto &i : covIndexMap) {
      auto covData = coverages.getScalarData(i.first);
      std::memcpy(flattenedCoverages.data() + i.second * covSize,
                  covData->data(), covSize * sizeof(NumericType));
    }
    model->getSurfaceModel()->getCoverages().alloc_and_upload(
        flattenedCoverages);
  }

  void moveCoveragesToTopLS(psDomainType domain,
                            psSmartPointer<translatorType> translator,
                            psPointData<NumericType> &coverages) {
    auto topLS = domain->getLevelSets()->back();
    for (size_t i = 0; i < coverages.getScalarDataSize(); i++) {
      auto covName = coverages.getScalarDataLabel(i);
      std::vector<NumericType> levelSetData(topLS->getNumberOfPoints(), 0);
      auto cov = coverages.getScalarData(covName);
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

  void
  updateCoveragesFromAdvectedSurface(psDomainType domain,
                                     psSmartPointer<translatorType> translator,
                                     psPointData<NumericType> &coverages) {
    auto topLS = domain->getLevelSets()->back();
    for (size_t i = 0; i < coverages.getScalarDataSize(); i++) {
      auto covName = coverages.getScalarDataLabel(i);
      auto levelSetData = topLS->getPointData().getScalarData(covName);
      auto covData = coverages.getScalarData(covName);
      covData->resize(translator->size());
      for (const auto it : *translator.get()) {
        covData->at(it.second) = levelSetData->at(it.first);
      }
    }
  }

  pscuContext_t *context;
  curtTracer<NumericType, D> rayTrace = curtTracer<NumericType, D>(context);

  psDomainType domain = nullptr;
  psSmartPointer<pscuProcessModel<NumericType>> model = nullptr;
  lsIntegrationSchemeEnum integrationScheme =
      lsIntegrationSchemeEnum::ENGQUIST_OSHER_1ST_ORDER;
  NumericType processDuration;
  long raysPerPoint = 1000;
  bool useRandomSeeds = true;
  size_t maxIterations = 20;
  bool periodicBoundary = false;
  bool coveragesInitialized = false;
  bool rayTracerInitialized = false;
  NumericType printTime = 0.;
  NumericType processTime = 0;
};
