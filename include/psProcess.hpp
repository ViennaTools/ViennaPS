#ifndef PS_PROCESS
#define PS_PROCESS

#include <lsAdvect.hpp>
#include <lsDomain.hpp>
#include <lsMesh.hpp>
#include <lsMessage.hpp>
#include <lsToDiskMesh.hpp>

#include <psDomain.hpp>
#include <psProcessModel.hpp>
#include <psSmartPointer.hpp>
#include <psSurfaceModel.hpp>
#include <psTranslationField.hpp>
#include <psVelocityField.hpp>
#include <psVolumeModel.hpp>

#include <rayBoundCondition.hpp>
#include <rayParticle.hpp>
#include <rayTrace.hpp>

template <typename NumericType, int D> class psProcess {
  using translatorType = std::unordered_map<unsigned long, unsigned long>;
  using psDomainType = psSmartPointer<psDomain<NumericType, D>>;

public:
  template <typename ProcessModelType>
  void setProcessModel(psSmartPointer<ProcessModelType> passedProcessModel) {
    model = std::dynamic_pointer_cast<psProcessModel<NumericType, D>>(
        passedProcessModel);
  }

  void setDomain(psSmartPointer<psDomain<NumericType, D>> passedDomain) {
    domain = passedDomain;
  }

  /// Set the source direction, where the rays should be traced from.
  void setSourceDirection(const rayTraceDirection passedDirection) {
    sourceDirection = passedDirection;
  }

  void setProcessDuration(double passedDuration) {
    processDuration = passedDuration;
  }

  void setNumberOfRaysPerPoint(long numRays) { raysPerPoint = numRays; }

  void setMaxCoverageInitIterations(size_t maxIt) { maxIterations = maxIt; }

  void apply() {
    /* ---------- Process Setup --------- */
    auto name = model->getProcessName();

    if (!domain) {
      lsMessage::getInstance()
          .addWarning("No domain passed to psProcess.")
          .print();
      return;
    }

    if (model->getGeometricModel()) {
      model->getGeometricModel()->setDomain(domain);
#ifdef VIENNAPS_VERBOSE
      std::cout << "Applying geometric model..." << std::endl;
#endif
      model->getGeometricModel()->apply();
      return;
    }

    if (processDuration == 0.) {
      // apply only volume model
      if (model->getVolumeModel()) {
        model->getVolumeModel()->setDomain(domain);
        model->getVolumeModel()->applyPreAdvect(0);
      } else {
        lsMessage::getInstance()
            .addWarning("No volume model passed to psProcess.")
            .print();
      }
      return;
    }

    if (!model->getSurfaceModel()) {
      lsMessage::getInstance()
          .addWarning("No surface model passed to psProcess.")
          .print();
      return;
    }

    double remainingTime = processDuration;
    assert(domain->getLevelSets()->size() != 0 && "No level sets in domain.");
    const NumericType gridDelta =
        domain->getLevelSets()->back()->getGrid().getGridDelta();

    auto diskMesh = lsSmartPointer<lsMesh<NumericType>>::New();
    auto translator = lsSmartPointer<translatorType>::New();
    lsToDiskMesh<NumericType, D> meshConverter(diskMesh);
    meshConverter.setTranslator(translator);

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

    rayTraceBoundary rayBoundaryCondition[D];
    rayTrace<NumericType, D> rayTrace;

    if (useRayTracing) {
      // Map the domain boundary to the ray tracing boundaries
      for (unsigned i = 0; i < D; ++i)
        rayBoundaryCondition[i] = convertBoundaryCondition(
            domain->getGrid().getBoundaryConditions(i));

      rayTrace.setSourceDirection(sourceDirection);
      rayTrace.setNumberOfRaysPerPoint(raysPerPoint);
      rayTrace.setBoundaryConditions(rayBoundaryCondition);
      rayTrace.setUseRandomSeeds(useRandomSeeds);
      rayTrace.setCalculateFlux(false);
    }

    // Determine whether volume model is used
    const bool useVolumeModel = model->getVolumeModel() != nullptr;
    if (useVolumeModel) {
      assert(domain->getUseCellSet());
      model->getVolumeModel()->setDomain(domain);
    }

    // Determine whether there are process parameters used in ray tracing
    if (model->getSurfaceModel())
      model->getSurfaceModel()->initializeProcessParameters();
    const bool useProcessParams =
        model->getSurfaceModel()->getProcessParameters() != nullptr;

#ifdef VIENNAPS_VERBOSE
    if (useProcessParams)
      std::cout << "Using process parameters." << std::endl;
    if (useVolumeModel)
      std::cout << "Using volume model." << std::endl;
#endif

    bool useCoverages = false;

    // Initialize coverages
    meshConverter.apply();
    auto numPoints = diskMesh->getNodes().size();
    if (!coveragesInitialized)
      model->getSurfaceModel()->initializeCoverages(numPoints);
    if (model->getSurfaceModel()->getCoverages() != nullptr) {
      useCoverages = true;
#ifdef VIENNAPS_VERBOSE
      std::cout << "Using coverages." << std::endl;
#endif
      if (!coveragesInitialized) {
#ifdef VIENNAPS_VERBOSE
        std::cout << "Initializing coverages ... " << std::endl;
#endif
        auto points = diskMesh->getNodes();
        auto normals = *diskMesh->getCellData().getVectorData("Normals");
        auto materialIds =
            *diskMesh->getCellData().getScalarData("MaterialIds");
        rayTrace.setGeometry(points, normals, gridDelta);
        rayTrace.setMaterialIds(materialIds);

        for (size_t iterations = 0; iterations < maxIterations; iterations++) {
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
          rayTrace.setGlobalData(rayTraceCoverages);

          auto Rates = psSmartPointer<psPointData<NumericType>>::New();

          for (auto &particle : *model->getParticleTypes()) {
            rayTrace.setParticleType(particle);
            rayTrace.apply();

            // fill up rates vector with rates from this particle type
            auto numRates = particle->getRequiredLocalDataSize();
            auto &localData = rayTrace.getLocalData();
            for (int i = 0; i < numRates; ++i) {
              auto rate = std::move(localData.getVectorData(i));

              // normalize rates
              rayTrace.normalizeFlux(rate);

              Rates->insertNextScalarData(std::move(rate),
                                          localData.getVectorDataLabel(i));
            }
          }

          // move coverages back in the model
          moveRayDataToPointData(model->getSurfaceModel()->getCoverages(),
                                 rayTraceCoverages);
          model->getSurfaceModel()->updateCoverages(Rates);
          coveragesInitialized = true;
#ifdef VIENNAPS_VERBOSE
          auto coverages = model->getSurfaceModel()->getCoverages();
          for (size_t idx = 0; idx < coverages->getScalarDataSize(); idx++) {
            auto label = coverages->getScalarDataLabel(idx);
            diskMesh->getCellData().insertNextScalarData(
                *coverages->getScalarData(idx), label);
          }
          for (size_t idx = 0; idx < Rates->getScalarDataSize(); idx++) {
            auto label = Rates->getScalarDataLabel(idx);
            diskMesh->getCellData().insertNextScalarData(
                *Rates->getScalarData(idx), label);
          }
          printDiskMesh(diskMesh, name + "_covIinit_" +
                                      std::to_string(iterations) + ".vtp");
          std::cerr << "\r"
                    << "Iteration: " << iterations + 1 << " / "
                    << maxIterations;
          if (iterations == maxIterations - 1)
            std::cerr << std::endl;
#endif
        }
      }
    }

    size_t counter = 0;
    while (remainingTime > 0.) {
#ifdef VIENNAPS_VERBOSE
      std::cout << "Remaining time: " << remainingTime << std::endl;
#endif

      auto Rates = psSmartPointer<psPointData<NumericType>>::New();
      meshConverter.apply();
      auto materialIds = *diskMesh->getCellData().getScalarData("MaterialIds");

      if (useRayTracing) {
        auto points = diskMesh->getNodes();
        auto normals = *diskMesh->getCellData().getVectorData("Normals");
        rayTrace.setGeometry(points, normals, gridDelta);
        rayTrace.setMaterialIds(materialIds);

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
          rayTrace.setGlobalData(rayTraceCoverages);
        }

        for (auto &particle : *model->getParticleTypes()) {
          rayTrace.setParticleType(particle);
          rayTrace.apply();

          // fill up rates vector with rates from this particle type
          auto numRates = particle->getRequiredLocalDataSize();
          auto &localData = rayTrace.getLocalData();
          for (int i = 0; i < numRates; ++i) {
            auto rate = std::move(localData.getVectorData(i));

            // normalize rates
            rayTrace.normalizeFlux(rate);
            Rates->insertNextScalarData(std::move(rate),
                                        localData.getVectorDataLabel(i));
          }
        }

        // move coverages back to model
        if (useCoverages)
          moveRayDataToPointData(model->getSurfaceModel()->getCoverages(),
                                 rayTraceCoverages);
      }

      auto velocitites =
          model->getSurfaceModel()->calculateVelocities(Rates, materialIds);
      model->getVelocityField()->setVelocities(velocitites);

#ifdef VIENNAPS_VERBOSE
      if (velocitites)
        diskMesh->getCellData().insertNextScalarData(*velocitites,
                                                     "velocities");
      if (useCoverages) {
        auto coverages = model->getSurfaceModel()->getCoverages();
        for (size_t idx = 0; idx < coverages->getScalarDataSize(); idx++) {
          auto label = coverages->getScalarDataLabel(idx);
          diskMesh->getCellData().insertNextScalarData(
              *coverages->getScalarData(idx), label);
        }
      }
      for (size_t idx = 0; idx < Rates->getScalarDataSize(); idx++) {
        auto label = Rates->getScalarDataLabel(idx);
        diskMesh->getCellData().insertNextScalarData(*Rates->getScalarData(idx),
                                                     label);
      }
      printDiskMesh(diskMesh, name + "_" + std::to_string(counter++) + ".vtp");
#endif
      // apply volume model
      if (useVolumeModel) {
        model->getVolumeModel()->applyPreAdvect(processDuration -
                                                remainingTime);
      }

      // move coverages to LS, so they get are moved with the advection step
      if (useCoverages)
        moveCoveragesToTopLS(translator,
                             model->getSurfaceModel()->getCoverages());
      advectionKernel.apply();

      // update the translator to retrieve the correct coverages from the LS
      meshConverter.apply();
      if (useCoverages)
        updateCoveragesFromAdvectedSurface(
            translator, model->getSurfaceModel()->getCoverages());

      // apply volume model
      if (useVolumeModel) {
        domain->getCellSet()->updateSurface();
        model->getVolumeModel()->applyPostAdvect(
            advectionKernel.getAdvectedTime());
      }

      remainingTime -= advectionKernel.getAdvectedTime();
    }

    addMaterialIdsToTopLS(translator,
                          diskMesh->getCellData().getScalarData("MaterialIds"));
  }

private:
  void printSurfaceMesh(lsSmartPointer<lsDomain<NumericType, D>> dom,
                        std::string name) {
    auto mesh = lsSmartPointer<lsMesh<NumericType>>::New();
    lsToSurfaceMesh<NumericType, D>(dom, mesh).apply();
    lsVTKWriter<NumericType>(mesh, name).apply();
  }

  void printDiskMesh(lsSmartPointer<lsMesh<NumericType>> mesh,
                     std::string name) {
    lsVTKWriter<NumericType>(mesh, name).apply();
  }

  rayTraceBoundary convertBoundaryCondition(
      lsBoundaryConditionEnum<D> originalBoundaryCondition) {
    switch (originalBoundaryCondition) {
    case lsBoundaryConditionEnum<D>::REFLECTIVE_BOUNDARY:
      return rayTraceBoundary::REFLECTIVE;

    case lsBoundaryConditionEnum<D>::INFINITE_BOUNDARY:
      return rayTraceBoundary::IGNORE;

    case lsBoundaryConditionEnum<D>::PERIODIC_BOUNDARY:
      return rayTraceBoundary::PERIODIC;

    case lsBoundaryConditionEnum<D>::POS_INFINITE_BOUNDARY:
      return rayTraceBoundary::IGNORE;

    case lsBoundaryConditionEnum<D>::NEG_INFINITE_BOUNDARY:
      return rayTraceBoundary::IGNORE;
    }
    return rayTraceBoundary::IGNORE;
  }

  rayTracingData<NumericType>
  movePointDataToRayData(psSmartPointer<psPointData<NumericType>> pointData) {
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
                         rayTracingData<NumericType> &rayData) {
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

  void addMaterialIdsToTopLS(lsSmartPointer<translatorType> translator,
                             std::vector<NumericType> *materialIds) {
    auto topLS = domain->getLevelSets()->back();
    std::vector<NumericType> levelSetData(topLS->getNumberOfPoints(), 0);
    for (const auto iter : *translator.get()) {
      levelSetData[iter.first] = materialIds->at(iter.second);
    }
    topLS->getPointData().insertNextScalarData(std::move(levelSetData),
                                               "Material");
  }

  void updateCoveragesFromAdvectedSurface(
      lsSmartPointer<translatorType> translator,
      psSmartPointer<psPointData<NumericType>> coverages) {
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

  psDomainType domain = nullptr;
  psSmartPointer<psProcessModel<NumericType, D>> model = nullptr;
  NumericType processDuration;
  rayTraceDirection sourceDirection =
      D == 3 ? rayTraceDirection::POS_Z : rayTraceDirection::POS_Y;
  long raysPerPoint = 1000;
  bool useRandomSeeds = true;
  size_t maxIterations = 20;
  bool coveragesInitialized = false;
};

#endif