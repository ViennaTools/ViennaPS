#ifndef PS_PROCESS
#define PS_PROCESS

#include <lsAdvect.hpp>
#include <lsDomain.hpp>
#include <lsMesh.hpp>
#include <lsToDiskMesh.hpp>

#include <psAdvectionCallback.hpp>
#include <psDomain.hpp>
#include <psLogger.hpp>
#include <psProcessModel.hpp>
#include <psSmartPointer.hpp>
#include <psSurfaceModel.hpp>
#include <psTranslationField.hpp>
#include <psVelocityField.hpp>

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

  void setProcessDuration(NumericType passedDuration) {
    processDuration = passedDuration;
  }

  NumericType getProcessDuration() const { return processTime; }

  void setNumberOfRaysPerPoint(long numRays) { raysPerPoint = numRays; }

  void setMaxCoverageInitIterations(size_t maxIt) { maxIterations = maxIt; }

  void setSmoothFlux(bool pSmoothFlux) { smoothFlux = pSmoothFlux; }

  void
  setIntegrationScheme(const lsIntegrationSchemeEnum passedIntegrationScheme) {
    integrationScheme = passedIntegrationScheme;
  }

<<<<<<< HEAD
  void setPrintIntermediate(const bool passedPrint) {
    printIntermediate = passedPrint;
=======
  // Sets the minimum time between printing intermediate results during the
  // process. If this is set to a non-positive value, no intermediate results
  // are printed.
  void setPrintTimeInterval(const NumericType passedTime) {
    printTime = passedTime;
>>>>>>> master
  }

  void apply() {
    /* ---------- Process Setup --------- */
    if (!model) {
      psLogger::getInstance()
          .addWarning("No process model passed to psProcess.")
          .print();
      return;
    }
    auto name = model->getProcessName();

<<<<<<< HEAD
    if (!domain) {
      lsMessage::getInstance()
=======
    if (!domain) {
      psLogger::getInstance()
>>>>>>> master
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
<<<<<<< HEAD
      } else {
        lsMessage::getInstance()
=======
      } else {
        psLogger::getInstance()
>>>>>>> master
            .addWarning("No advection callback passed to psProcess.")
            .print();
      }
      return;
    }

<<<<<<< HEAD
    if (!model->getSurfaceModel()) {
      lsMessage::getInstance()
=======
    if (!model->getSurfaceModel()) {
      psLogger::getInstance()
>>>>>>> master
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

      // initialize particle data logs
      particleDataLogs.resize(model->getParticleTypes()->size());
      for (std::size_t i = 0; i < model->getParticleTypes()->size(); i++) {
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
    if (!coveragesInitialized)
      model->getSurfaceModel()->initializeCoverages(numPoints);
<<<<<<< HEAD
    if (model->getSurfaceModel()->getCoverages() != nullptr) {
      useCoverages = true;
#ifdef VIENNAPS_VERBOSE
      std::cout << "Using coverages." << std::endl;
#endif
      if (!coveragesInitialized) {
#ifdef VIENNAPS_VERBOSE
        std::cout << "Initializing coverages ... " << std::endl;
#endif
        ======= if (model->getSurfaceModel()->getCoverages() != nullptr) {
          psUtils::Timer timer;
          useCoverages = true;
          psLogger::getInstance().addInfo("Using coverages.").print();
          if (!coveragesInitialized) {
            timer.start();
            psLogger::getInstance()
                .addInfo("Initializing coverages ... ")
                .print();
>>>>>>> master
            auto points = diskMesh->getNodes();
            auto normals = *diskMesh->getCellData().getVectorData("Normals");
            auto materialIds =
                *diskMesh->getCellData().getScalarData("MaterialIds");
            rayTrace.setGeometry(points, normals, gridDelta);
            rayTrace.setMaterialIds(materialIds);

            for (size_t iterations = 0; iterations < maxIterations;
                 iterations++) {
              // move coverages to the ray tracer
              rayTracingData<NumericType> rayTraceCoverages =
                  movePointDataToRayData(
                      model->getSurfaceModel()->getCoverages());
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

<<<<<<< HEAD
              for (auto &particle : *model->getParticleTypes()) {
=======
      std::size_t particleIdx = 0;
      for (auto &particle : *model->getParticleTypes()) {
        int dataLogSize = model->getParticleLogSize(particleIdx);
        if (dataLogSize > 0) {
          rayTrace.getDataLog().data.resize(1);
          rayTrace.getDataLog().data[0].resize(dataLogSize, 0.);
        }
>>>>>>> master
                rayTrace.setParticleType(particle);
                rayTrace.apply();

                // fill up rates vector with rates from this particle type
                auto &localData = rayTrace.getLocalData();
<<<<<<< HEAD
                for (int i = 0; i < numRates; ++i) {
=======
        for (int i = 0; i < particle->getRequiredLocalDataSize(); ++i) {
>>>>>>> master
                  auto rate = std::move(localData.getVectorData(i));

                  // normalize fluxes
                  rayTrace.normalizeFlux(rate);
                  if (smoothFlux)
                    rayTrace.smoothFlux(rate);
                  Rates->insertNextScalarData(std::move(rate),
                                              localData.getVectorDataLabel(i));
                }

                if (dataLogSize > 0) {
                  particleDataLogs[particleIdx].merge(rayTrace.getDataLog());
                }
                ++particleIdx;
              }

              // move coverages back in the model
              moveRayDataToPointData(model->getSurfaceModel()->getCoverages(),
                                     rayTraceCoverages);
              model->getSurfaceModel()->updateCoverages(Rates);
              coveragesInitialized = true;
<<<<<<< HEAD
              if (printIntermediate) {
                auto coverages = model->getSurfaceModel()->getCoverages();
                for (size_t idx = 0; idx < coverages->getScalarDataSize();
                     idx++) {
=======

      if (psLogger::getLogLevel() >= 3) {
        auto coverages = model->getSurfaceModel()->getCoverages();
        for (size_t idx = 0; idx < coverages->getScalarDataSize(); idx++) {
>>>>>>> master
                  auto label = coverages->getScalarDataLabel(idx);
                  diskMesh->getCellData().insertNextScalarData(
                      *coverages->getScalarData(idx), label);
                }
<<<<<<< HEAD
                for (size_t idx = 0; idx < Rates->getScalarDataSize(); idx++) {
=======
        for (size_t idx = 0; idx < Rates->getScalarDataSize(); idx++) {
>>>>>>> master
                  auto label = Rates->getScalarDataLabel(idx);
                  diskMesh->getCellData().insertNextScalarData(
                      *Rates->getScalarData(idx), label);
                }
                printDiskMesh(diskMesh, name + "_covIinit_" +
                                            std::to_string(iterations) +
                                            ".vtp");
<<<<<<< HEAD
              }
#ifdef VIENNAPS_VERBOSE
              std::cerr << "\r"
                        << "Iteration: " << iterations + 1 << " / "
                        << maxIterations;
              if (iterations == maxIterations - 1)
                std::cerr << std::endl;
#endif
              ======= psLogger::getInstance()
                  .addInfo("Iteration: " + std::to_string(iterations))
                  .print();
            }
>>>>>>> master
          }
          timer.finish();
          psLogger::getInstance()
              .addTiming("Coverage initialization", timer)
              .print();
        }
      }

      double previousTimeStep = 0.;
      size_t counter = 0;
<<<<<<< HEAD
      while (remainingTime > 0.) {
#ifdef VIENNAPS_VERBOSE
        std::cout << "Remaining time: " << remainingTime << std::endl;
#endif
        ======= psUtils::Timer rtTimer;
        psUtils::Timer callbackTimer;
        psUtils::Timer advTimer;
        while (remainingTime > 0.) {
          psLogger::getInstance()
              .addInfo("Remaining time: " + std::to_string(remainingTime))
              .print();
>>>>>>> master

          auto Rates = psSmartPointer<psPointData<NumericType>>::New();
          meshConverter.apply();
          auto materialIds =
              *diskMesh->getCellData().getScalarData("MaterialIds");
          auto points = diskMesh->getNodes();

<<<<<<< HEAD
          if (useRayTracing) {
            auto points = diskMesh->getNodes();
=======
  // rate calculation by top-down ray tracing
  if (useRayTracing) {
    rtTimer.start();
>>>>>>> master
            auto normals = *diskMesh->getCellData().getVectorData("Normals");
            rayTrace.setGeometry(points, normals, gridDelta);
            rayTrace.setMaterialIds(materialIds);

            // move coverages to ray tracer
            rayTracingData<NumericType> rayTraceCoverages;
            if (useCoverages) {
              rayTraceCoverages = movePointDataToRayData(
                  model->getSurfaceModel()->getCoverages());
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

<<<<<<< HEAD
            for (auto &particle : *model->getParticleTypes()) {
=======
    std::size_t particleIdx = 0;
    for (auto &particle : *model->getParticleTypes()) {
      int dataLogSize = model->getParticleLogSize(particleIdx);
      if (dataLogSize > 0) {
        rayTrace.getDataLog().data.resize(1);
        rayTrace.getDataLog().data[0].resize(dataLogSize, 0.);
      }
>>>>>>> master
              rayTrace.setParticleType(particle);
              rayTrace.apply();

              // fill up rates vector with rates from this particle type
              auto numRates = particle->getRequiredLocalDataSize();
              auto &localData = rayTrace.getLocalData();
              for (int i = 0; i < numRates; ++i) {
                auto rate = std::move(localData.getVectorData(i));

                // normalize rates
                rayTrace.normalizeFlux(rate);
                if (smoothFlux)
                  rayTrace.smoothFlux(rate);
                Rates->insertNextScalarData(std::move(rate),
                                            localData.getVectorDataLabel(i));
              }

              if (dataLogSize > 0) {
                particleDataLogs[particleIdx].merge(rayTrace.getDataLog());
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
          auto velocitites = model->getSurfaceModel()->calculateVelocities(
              Rates, points, materialIds);
          model->getVelocityField()->setVelocities(velocitites);
          if (model->getVelocityField()->getTranslationFieldOptions() == 2)
            transField->buildKdTree(points);

<<<<<<< HEAD
          if (printIntermediate) {
            if (velocitites)
              diskMesh->getCellData().insertNextScalarData(*velocitites,
                                                           "velocities");
            if (useCoverages) {
              auto coverages = model->getSurfaceModel()->getCoverages();
              for (size_t idx = 0; idx < coverages->getScalarDataSize();
                   idx++) {
=======
  // print debug output
  if (psLogger::getLogLevel() >= 4) {
    if (velocitites)
      diskMesh->getCellData().insertNextScalarData(*velocitites, "velocities");
    if (useCoverages) {
      auto coverages = model->getSurfaceModel()->getCoverages();
      for (size_t idx = 0; idx < coverages->getScalarDataSize(); idx++) {
>>>>>>> master
                auto label = coverages->getScalarDataLabel(idx);
                diskMesh->getCellData().insertNextScalarData(
                    *coverages->getScalarData(idx), label);
              }
            }
<<<<<<< HEAD
            for (size_t idx = 0; idx < Rates->getScalarDataSize(); idx++) {
=======
    for (size_t idx = 0; idx < Rates->getScalarDataSize(); idx++) {
>>>>>>> master
              auto label = Rates->getScalarDataLabel(idx);
              diskMesh->getCellData().insertNextScalarData(
                  *Rates->getScalarData(idx), label);
            }
<<<<<<< HEAD
            printDiskMesh(diskMesh,
                          name + "_" + std::to_string(counter++) + ".vtp");
#endif
            // apply advection callback
            if (useAdvectionCallback) {
              model->getAdvectionCallback()->applyPreAdvect(processDuration -
                                                            remainingTime);
            }

            // move coverages to LS, so they get are moved with the advection
            // step
            if (useCoverages)
              moveCoveragesToTopLS(translator,
                                   model->getSurfaceModel()->getCoverages());
            advectionKernel.apply();

            // update the translator to retrieve the correct coverages from the
            // LS
            meshConverter.apply();
            if (useCoverages)
              updateCoveragesFromAdvectedSurface(
                  translator, model->getSurfaceModel()->getCoverages());

            // apply advection callback
            if (useAdvectionCallback) {
              if (domain->getUseCellSet())
                domain->getCellSet()->updateSurface();
              model->getAdvectionCallback()->applyPostAdvect(
                  advectionKernel.getAdvectedTime());
            }

            remainingTime -= advectionKernel.getAdvectedTime();
          }

          addMaterialIdsToTopLS(
              translator, diskMesh->getCellData().getScalarData("MaterialIds"));
        }
=======
if (printTime >= 0. &&
    ((processDuration - remainingTime) - printTime * counter) > -1.) {
  printDiskMesh(diskMesh, name + "_" + std::to_string(counter) + ".vtp");
  if (domain->getUseCellSet()) {
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
  moveCoveragesToTopLS(translator, model->getSurfaceModel()->getCoverages());
advTimer.start();
advectionKernel.apply();
advTimer.finish();
psLogger::getInstance().addTiming("Surface advection", advTimer).print();
>>>>>>> master

      private:
        void printSurfaceMesh(lsSmartPointer<lsDomain<NumericType, D>> dom,
                              std::string name) {
          auto mesh = lsSmartPointer<lsMesh<NumericType>>::New();
          lsToSurfaceMesh<NumericType, D>(dom, mesh).apply();
          lsVTKWriter<NumericType>(mesh, name).apply();
        }

<<<<<<< HEAD
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

        rayTracingData<NumericType> movePointDataToRayData(
            psSmartPointer<psPointData<NumericType>> pointData) {
          rayTracingData<NumericType> rayData;
          const auto numData = pointData->getScalarDataSize();
          rayData.setNumberOfVectorData(numData);
          for (size_t i = 0; i < numData; ++i) {
            auto label = pointData->getScalarDataLabel(i);
            rayData.setVectorData(
                i, std::move(*pointData->getScalarData(label)), label);
          }

          return std::move(rayData);
        }

        void moveRayDataToPointData(
            psSmartPointer<psPointData<NumericType>> pointData,
            rayTracingData<NumericType> & rayData) {
          pointData->clear();
          const auto numData = rayData.getVectorData().size();
          for (size_t i = 0; i < numData; ++i)
            pointData->insertNextScalarData(std::move(rayData.getVectorData(i)),
                                            rayData.getVectorDataLabel(i));
=======
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
}

processTime = processDuration - remainingTime;
processTimer.finish();

psLogger::getInstance()
    .addTiming("\nProcess " + name, processTimer)
    .addTiming("Surface advection total time", advTimer.totalDuration * 1e-9,
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
void printSurfaceMesh(lsSmartPointer<lsDomain<NumericType, D>> dom,
                      std::string name) {
  auto mesh = lsSmartPointer<lsMesh<NumericType>>::New();
  lsToSurfaceMesh<NumericType, D>(dom, mesh).apply();
  psVTKWriter<NumericType>(mesh, name).apply();
}

void printDiskMesh(lsSmartPointer<lsMesh<NumericType>> mesh, std::string name) {
  psVTKWriter<NumericType>(mesh, name).apply();
}

rayTraceBoundary
convertBoundaryCondition(lsBoundaryConditionEnum<D> originalBoundaryCondition) {
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
>>>>>>> master
        }

        void moveCoveragesToTopLS(
            lsSmartPointer<translatorType> translator,
            psSmartPointer<psPointData<NumericType>> coverages) {
          auto topLS = domain->getLevelSets()->back();
          for (size_t i = 0; i < coverages->getScalarDataSize(); i++) {
            auto covName = coverages->getScalarDataLabel(i);
            std::vector<NumericType> levelSetData(topLS->getNumberOfPoints(),
                                                  0);
            auto cov = coverages->getScalarData(covName);
            for (const auto iter : *translator.get()) {
              levelSetData[iter.first] = cov->at(iter.second);
            }
            if (auto data = topLS->getPointData().getScalarData(covName);
                data != nullptr) {
              *data = std::move(levelSetData);
            } else {
              topLS->getPointData().insertNextScalarData(
                  std::move(levelSetData), covName);
            }
          }
        }

        void addMaterialIdsToTopLS(lsSmartPointer<translatorType> translator,
                                   std::vector<NumericType> * materialIds) {
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

<<<<<<< HEAD
        psDomainType domain = nullptr;
        psSmartPointer<psProcessModel<NumericType, D>> model = nullptr;
        NumericType processDuration;
        rayTraceDirection sourceDirection =
            D == 3 ? rayTraceDirection::POS_Z : rayTraceDirection::POS_Y;
        lsIntegrationSchemeEnum integrationScheme =
            lsIntegrationSchemeEnum::ENGQUIST_OSHER_1ST_ORDER;
        long raysPerPoint = 1000;
        bool useRandomSeeds = true;
        size_t maxIterations = 20;
        bool coveragesInitialized = false;
        bool printIntermediate = true;
      };
=======
  void addMaterialIdsToTopLS(lsSmartPointer<translatorType> translator,
                             std::vector<NumericType> * materialIds) {
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
  psSmartPointer<psMaterialMap> materialMap = nullptr;
  NumericType processDuration;
  rayTraceDirection sourceDirection =
      D == 3 ? rayTraceDirection::POS_Z : rayTraceDirection::POS_Y;
  lsIntegrationSchemeEnum integrationScheme =
      lsIntegrationSchemeEnum::ENGQUIST_OSHER_1ST_ORDER;
  long raysPerPoint = 1000;
  std::vector<rayDataLog<NumericType>> particleDataLogs;
  bool useRandomSeeds = true;
  bool smoothFlux = false;
  size_t maxIterations = 20;
  bool coveragesInitialized = false;
  NumericType printTime = 0.;
  NumericType processTime = 0.;
};
>>>>>>> master

#endif
