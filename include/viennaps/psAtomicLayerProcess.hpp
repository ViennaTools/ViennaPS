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

template <typename NumericType, int D> class psAtomicLayerProcess {
  using translatorType = std::unordered_map<unsigned long, unsigned long>;
  using psDomainType = psSmartPointer<psDomain<NumericType, D>>;

public:
  psAtomicLayerProcess() {}
  psAtomicLayerProcess(psDomainType domain) : pDomain_(domain) {}
  // Constructor for a process with a pre-configured process model.
  template <typename ProcessModelType>
  psAtomicLayerProcess(psDomainType domain,
                       psSmartPointer<ProcessModelType> passedProcessModel)
      : pDomain_(domain), pModel_() {
    pModel_ = std::dynamic_pointer_cast<psProcessModel<NumericType, D>>(
        passedProcessModel);
  }

  // Set the process model. This can be either a pre-configured process model or
  // a custom process model. A custom process model must interface the
  // psProcessModel class.
  template <typename ProcessModelType>
  void setProcessModel(psSmartPointer<ProcessModelType> passedProcessModel) {
    pModel_ = std::dynamic_pointer_cast<psProcessModel<NumericType, D>>(
        passedProcessModel);
  }

  // Set the process domain.
  void setDomain(psDomainType domain) { pDomain_ = domain; }

  // Set the source direction, where the rays should be traced from.
  void setSourceDirection(rayTraceDirection passedDirection) {
    sourceDirection_ = passedDirection;
  }

  // Set the duration of the process.
  void setMeanFreePath(NumericType lambda) { lambda_ = lambda; }

  void setPulseTime(NumericType pulseTime) { pulseTime_ = pulseTime; }

  void setCoverageTimeStep(NumericType coverageTimeStep) {
    coverageTimeStep_ = coverageTimeStep;
  }

  void setNumCycles(unsigned int numCycles) { numCycles_ = numCycles; }

  // Specify the number of rays to be traced for each particle throughout the
  // process. The total count of rays is the product of this number and the
  // number of points in the process geometry.
  void setNumberOfRaysPerPoint(unsigned numRays) { raysPerPoint_ = numRays; }

  // Set the integration scheme for solving the level-set equation.
  // Possible integration schemes are specified in lsIntegrationSchemeEnum.
  void setIntegrationScheme(lsIntegrationSchemeEnum passedIntegrationScheme) {
    integrationScheme_ = passedIntegrationScheme;
  }

  // Enable the use of random seeds for ray tracing. This is useful to
  // prevent the formation of artifacts in the flux calculation.
  void enableRandomSeeds() { useRandomSeeds_ = true; }

  // Disable the use of random seeds for ray tracing.
  void disableRandomSeeds() { useRandomSeeds_ = false; }

  // Run the process.
  void apply() {

    checkInput();

    /* ---------- Process Setup --------- */
    psUtils::Timer processTimer;
    processTimer.start();

    auto name = pModel_->getProcessName().value_or("default");

    const NumericType gridDelta = pDomain_->getGrid().getGridDelta();
    auto diskMesh = lsSmartPointer<lsMesh<NumericType>>::New();
    auto translator = lsSmartPointer<translatorType>::New();
    lsToDiskMesh<NumericType, D> meshConverter(diskMesh);
    meshConverter.setTranslator(translator);
    if (pDomain_->getMaterialMap() && pDomain_->getMaterialMap()->size() ==
                                          pDomain_->getLevelSets()->size()) {
      meshConverter.setMaterialMap(
          pDomain_->getMaterialMap()->getMaterialMap());
    }

    auto transField = psSmartPointer<psTranslationField<NumericType>>::New(
        pModel_->getVelocityField(), pDomain_->getMaterialMap());
    transField->setTranslator(translator);

    lsAdvect<NumericType, D> advectionKernel;
    advectionKernel.setVelocityField(transField);
    advectionKernel.setIntegrationScheme(integrationScheme_);
    advectionKernel.setAdvectionTime(1.);

    for (auto dom : *pDomain_->getLevelSets()) {
      meshConverter.insertNextLevelSet(dom);
      advectionKernel.insertNextLevelSet(dom);
    }

    /* --------- Setup for ray tracing ----------- */

    rayBoundaryCondition rayBoundaryCondition[D];
    rayTrace<NumericType, D> rayTracer;

    // Map the domain boundary to the ray tracing boundaries
    for (unsigned i = 0; i < D; ++i)
      rayBoundaryCondition[i] = psUtils::convertBoundaryCondition<D>(
          pDomain_->getGrid().getBoundaryConditions(i));

    rayTracer.setSourceDirection(sourceDirection_);
    rayTracer.setNumberOfRaysPerPoint(raysPerPoint_);
    rayTracer.setBoundaryConditions(rayBoundaryCondition);
    rayTracer.setUseRandomSeeds(useRandomSeeds_);
    auto primaryDirection = pModel_->getPrimaryDirection();
    if (primaryDirection) {
      psLogger::getInstance()
          .addInfo("Using primary direction: " +
                   psUtils::arrayToString(primaryDirection.value()))
          .print();
      rayTracer.setPrimaryDirection(primaryDirection.value());
    }
    rayTracer.setCalculateFlux(false);
    rayTracer.setMeanFreePath(lambda_);

    // initialize particle data logs
    particleDataLogs_.resize(pModel_->getParticleTypes()->size());
    for (std::size_t i = 0; i < pModel_->getParticleTypes()->size(); i++) {
      int logSize = pModel_->getParticleLogSize(i);
      if (logSize > 0) {
        particleDataLogs_[i].data.resize(1);
        particleDataLogs_[i].data[0].resize(logSize);
      }
    }

    auto surfaceModel = pModel_->getSurfaceModel();

    // Determine whether there are process parameters used in ray tracing
    surfaceModel->initializeProcessParameters();
    const bool useProcessParams =
        surfaceModel->getProcessParameters() != nullptr;
    if (useProcessParams)
      psLogger::getInstance().addInfo("Using process parameters.").print();

    size_t counter = 0;
    int numCycles = 0;
    while (numCycles++ < numCycles_) {
      psLogger::getInstance()
          .addInfo("Cycle: " + std::to_string(numCycles) + "/" +
                   std::to_string(numCycles_))
          .print();

      meshConverter.apply();
      auto numPoints = diskMesh->nodes.size();
      surfaceModel->initializeCoverages(numPoints);
      auto rates = psSmartPointer<psPointData<NumericType>>::New();
      auto const materialIds =
          *diskMesh->getCellData().getScalarData("MaterialIds");
      auto const points = diskMesh->getNodes();

      auto normals = *diskMesh->getCellData().getVectorData("Normals");
      rayTracer.setGeometry(points, normals, gridDelta);
      rayTracer.setMaterialIds(materialIds);

      NumericType time = 0.;
      int pulseCounter = 0;
      while (time < pulseTime_) {
#ifdef VIENNAPS_PYTHON_BUILD
        if (PyErr_CheckSignals() != 0)
          throw pybind11::error_already_set();
#endif

        psLogger::getInstance()
            .addInfo("Pulse time: " + std::to_string(time) + "/" +
                     std::to_string(pulseTime_))
            .print();

        // move coverages to ray tracer
        auto rayTraceCoverages =
            movePointDataToRayData(surfaceModel->getCoverages());

        if (useProcessParams) {
          // store scalars in addition to coverages
          auto processParams = surfaceModel->getProcessParameters();
          NumericType numParams = processParams->getScalarData().size();
          rayTraceCoverages.setNumberOfScalarData(numParams);
          for (size_t i = 0; i < numParams; ++i) {
            rayTraceCoverages.setScalarData(
                i, processParams->getScalarData(i),
                processParams->getScalarDataLabel(i));
          }
        }
        rayTracer.setGlobalData(rayTraceCoverages);

        rates->clear();
        std::size_t particleIdx = 0;
        for (auto &particle : *pModel_->getParticleTypes()) {
          int dataLogSize = pModel_->getParticleLogSize(particleIdx);
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
            rayTracer.smoothFlux(rate);
            rates->insertNextScalarData(std::move(rate),
                                        localData.getVectorDataLabel(i));
          }

          if (dataLogSize > 0) {
            particleDataLogs_[particleIdx].merge(rayTracer.getDataLog());
          }
          ++particleIdx;
        }

        // move coverages back to model
        moveRayDataToPointData(surfaceModel->getCoverages(), rayTraceCoverages);
        surfaceModel->updateCoverages(rates, materialIds);

        // print debug output
        if (psLogger::getLogLevel() >= 4) {
          for (size_t idx = 0; idx < rates->getScalarDataSize(); idx++) {
            auto label = rates->getScalarDataLabel(idx);
            diskMesh->getCellData().insertNextScalarData(
                *rates->getScalarData(idx), label);
          }
          auto coverages = surfaceModel->getCoverages();
          for (size_t idx = 0; idx < coverages->getScalarDataSize(); idx++) {
            auto label = coverages->getScalarDataLabel(idx);
            diskMesh->getCellData().insertNextScalarData(
                *coverages->getScalarData(idx), label);
          }
          printDiskMesh(diskMesh, name + "_pulse_" +
                                      std::to_string(pulseCounter++) + ".vtp");
        }

        time += coverageTimeStep_;
      }

      // get velocities
      auto velocities =
          surfaceModel->calculateVelocities(rates, points, materialIds);
      pModel_->getVelocityField()->setVelocities(velocities);
      if (pModel_->getVelocityField()->getTranslationFieldOptions() == 2)
        transField->buildKdTree(points);

      // print debug output
      if (psLogger::getLogLevel() >= 4) {
        diskMesh->getCellData().insertNextScalarData(*velocities, "velocities");
        printDiskMesh(diskMesh, name + "_" + std::to_string(counter) + ".vtp");
        counter++;
      }

      advectionKernel.apply();
    }

    processTimer.finish();

    psLogger::getInstance()
        .addTiming("\nProcess " + name, processTimer)
        .print();
  }

  void writeParticleDataLogs(std::string fileName) {
    std::ofstream file(fileName.c_str());

    for (std::size_t i = 0; i < particleDataLogs_.size(); i++) {
      if (!particleDataLogs_[i].data.empty()) {
        file << "particle" << i << "_data ";
        for (std::size_t j = 0; j < particleDataLogs_[i].data[0].size(); j++) {
          file << particleDataLogs_[i].data[0][j] << " ";
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

  void checkInput() const {
    if (!pDomain_) {
      psLogger::getInstance()
          .addError("No domain passed to psAtomicLayerProcess.")
          .print();
    }

    if (pDomain_->getLevelSets()->size() == 0) {
      psLogger::getInstance()
          .addError("No level sets in domain passed to psAtomicLayerProcess.")
          .print();
    }

    if (!pModel_) {
      psLogger::getInstance()
          .addError("No process model passed to psAtomicLayerProcess.")
          .print();
    }

    if (!pModel_->getSurfaceModel()) {
      psLogger::getInstance()
          .addError("No surface model passed to psAtomicLayerProcess.")
          .print();
    }

    if (!pModel_->getVelocityField()) {
      psLogger::getInstance()
          .addError("No velocity field passed to psAtomicLayerProcess.")
          .print();
    }

    if (!pModel_->getParticleTypes()) {
      psLogger::getInstance()
          .addError("No particle types specified for ray tracing.")
          .print();
    }

    if (pModel_->getGeometricModel()) {
      psLogger::getInstance()
          .addWarning("Geometric model not supported in ALP ...")
          .print();
    }

    if (pModel_->getAdvectionCallback()) {
      psLogger::getInstance()
          .addWarning("Advection callback not supported in ALP ...")
          .print();
    }
  }

  psDomainType pDomain_;
  psSmartPointer<psProcessModel<NumericType, D>> pModel_;

  rayTraceDirection sourceDirection_ =
      D == 3 ? rayTraceDirection::POS_Z : rayTraceDirection::POS_Y;
  lsIntegrationSchemeEnum integrationScheme_ =
      lsIntegrationSchemeEnum::ENGQUIST_OSHER_1ST_ORDER;
  unsigned raysPerPoint_ = 1000;
  bool useRandomSeeds_ = true;
  std::vector<rayDataLog<NumericType>> particleDataLogs_;

  unsigned int numCycles_ = 0;
  NumericType pulseTime_ = 0.;
  NumericType coverageTimeStep_ = 1.;
  NumericType lambda_ = -1.;
};