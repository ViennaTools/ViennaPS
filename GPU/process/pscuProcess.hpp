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
#include <pscuVolumeModel.hpp>

#include <psDomain.hpp>
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

  void setPrintIntermediate(const int passedPrint) {
    printIntermediate = static_cast<bool>(passedPrint);
  }

  void setPeriodicBoundary(const int passedPeriodic) {
    periodicBoundary = static_cast<bool>(passedPeriodic);
  }

  void apply() {
    /* ---------- Process Setup --------- */
    auto name = model->getProcessName();
    double remainingTime = processDuration;
    assert(domain->getLevelSets()->size() != 0 && "No level sets in domain.");
    const NumericType gridDelta =
        domain->getLevelSets()->back()->getGrid().getGridDelta();
    utCudaBuffer d_rates;

    auto diskMesh = psSmartPointer<lsMesh<NumericType>>::New();
    auto translator = psSmartPointer<translatorType>::New();
    lsToDiskMesh<NumericType, D> meshConverter(diskMesh);
    meshConverter.setTranslator(translator);

    assert(model->getVelocityField() != nullptr);
    auto transField = psSmartPointer<psTranslationField<NumericType>>::New();
    transField->setTranslator(translator);
    transField->setVelocityField(model->getVelocityField());

    lsAdvect<NumericType, D> advectionKernel;
    advectionKernel.setVelocityField(transField);

    for (auto dom : *domain->getLevelSets()) {
      meshConverter.insertNextLevelSet(dom);
      advectionKernel.insertNextLevelSet(dom);
    }

    /* --------- Setup for ray tracing ----------- */
    const bool useRayTracing = model->getParticleTypes() != nullptr;

    if (useRayTracing && !rayTracerInitialized) {
      if (!model->getPtxCode()) {
        std::cout << "No pipeline in process model. Aborting." << std::endl;
        return;
      }
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

    // Determine whether volume model is used
    const bool useVolumeModel = model->getVolumeModel() != nullptr;
    if (useVolumeModel) {
      assert(domain->getUseCellSet());
      model->getVolumeModel()->setDomain(domain);
    }

    // Determine whether there are process parameters used in ray tracing
    model->getSurfaceModel()->initializeProcessParameters();
    const bool useProcessParams =
        model->getSurfaceModel()->getProcessParameters().d_ptr != nullptr;

#ifdef VIENNAPS_VERBOSE
    if (useProcessParams)
      std::cout << "Using process parameters." << std::endl;
    if (useVolumeModel)
      std::cout << "Using volume model." << std::endl;
#endif

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

#ifdef VIENNAPS_VERBOSE
      std::cout << "Using coverages." << std::endl;
#endif

      if (!coveragesInitialized) {

#ifdef VIENNAPS_VERBOSE
        std::cout << "Initializing coverages ... " << std::endl;
#endif
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

#ifdef VIENNAPS_VERBOSE
          rayTrace.downloadResultsToPointData(diskMesh->getCellData(), d_rates,
                                              diskMesh->nodes.size());
          downloadCoverages(diskMesh->getCellData(),
                            model->getSurfaceModel()->getCoverages(),
                            diskMesh->nodes.size());
          printDiskMesh(diskMesh, name + "_covIinit_" +
                                      std::to_string(iterations) + ".vtp");
          std::cerr << "\r"
                    << "Iteration: " << iterations << " / " << maxIterations;
          if (iterations == maxIterations)
            std::cerr << std::endl;
#endif
        }
        coveragesInitialized = true;
      }
    }

    size_t counter = 0;
    size_t printCounter = 0;
    while (remainingTime > 0.) {
#ifdef VIENNAPS_VERBOSE
      std::cout << "Remaining time: " << remainingTime << std::endl;
#endif

      meshConverter.apply();
      auto materialIds = *diskMesh->getCellData().getScalarData("MaterialIds");

      if (useRayTracing) {
        // get coverages in ray tracer
        if (useCoverages)
          rayTrace.translateFromPointData(
              diskMesh, model->getSurfaceModel()->getCoverages(), numCov);
        rayTrace.apply();

        // get results as point data
        rayTrace.translateToPointData(diskMesh, d_rates);
      }

      auto velocitites =
          model->getSurfaceModel()->calculateVelocities(d_rates, materialIds);
      curtSmoothing<NumericType, D>(diskMesh, velocitites, gridDelta).apply();
      model->getVelocityField()->setVelocities(velocitites);

#ifdef VIENNAPS_VERBOSE
      if (printIntermediate) {
        rayTrace.downloadResultsToPointData(diskMesh->getCellData(), d_rates,
                                            diskMesh->nodes.size());
        if (useCoverages) {
          downloadCoverages(diskMesh->getCellData(),
                            model->getSurfaceModel()->getCoverages(),
                            diskMesh->nodes.size());
        }
        diskMesh->getCellData().insertNextScalarData(*velocitites, "etchRate");
        printDiskMesh(diskMesh,
                      name + "_point_" + std::to_string(counter++) + ".vtp");
      }
#endif

      // apply volume model
      if (useVolumeModel) {
        model->getVolumeModel()->applyPreAdvect(processDuration -
                                                remainingTime);
      }

      // move coverages to LS, so they get are moved with the advection step
      psPointData<NumericType> coverages;
      if (useCoverages) {
        downloadCoverages(coverages, model->getSurfaceModel()->getCoverages(),
                          diskMesh->nodes.size());
        moveCoveragesToTopLS(domain, translator, coverages);
      }

      // print intermediary time step
      if (printIntermediate) {

        auto mesh = psSmartPointer<lsMesh<NumericType>>::New();
        culsToSurfaceMesh<NumericType>(domain->getLevelSets()->back(), mesh)
            .apply();

        std::vector<NumericType> cellflux(mesh->triangles.size());
        rayTrace.getFlux(cellflux.data(), 0, 0);

        mesh->cellData.insertNextScalarData(cellflux, "flux");
        lsVTKWriter<NumericType>(
            mesh, name + "_step_" + std::to_string(printCounter) + ".vtp")
            .apply();

        // domain->printSurface(name + "_step_" + std::to_string(printCounter) +
        //                      ".vtp");
        if (useVolumeModel)
          domain->getCellSet()->writeVTU(name + "_step_" +
                                         std::to_string(printCounter) + ".vtu");
        printCounter++;
      }

      advectionKernel.apply();
      rayTrace.updateSurface();

      // update the translator to retrieve the correct coverages from the LS
      meshConverter.apply();
      if (useCoverages) {
        updateCoveragesFromAdvectedSurface(domain, translator, coverages);
        uploadCoverages(coverages);
      }

      // apply volume model
      if (useVolumeModel) {
        domain->getCellSet()->updateSurface();
        model->getVolumeModel()->applyPostAdvect(
            advectionKernel.getAdvectedTime());
      }

      remainingTime -= advectionKernel.getAdvectedTime();
    } // main process loop
    rayTrace.invalidateGeometry();
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
  NumericType processDuration;
  long raysPerPoint = 1000;
  bool useRandomSeeds = true;
  size_t maxIterations = 20;
  bool printIntermediate = false;
  bool periodicBoundary = false;
  bool coveragesInitialized = false;
  bool rayTracerInitialized = false;
};
