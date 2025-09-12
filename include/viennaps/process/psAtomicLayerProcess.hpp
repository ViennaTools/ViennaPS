#pragma once

#include "psDomain.hpp"
#include "psPreCompileMacros.hpp"
#include "psProcessModel.hpp"
#include "psProcessParams.hpp"
#include "psTranslationField.hpp"
#include "psUnits.hpp"
#include "psUtil.hpp"

#include <lsAdvect.hpp>
#include <lsDomain.hpp>
#include <lsMesh.hpp>
#include <lsToDiskMesh.hpp>

#include <rayReflection.hpp>
#include <raySource.hpp>
#include <rayTrace.hpp>

namespace viennaps {

using namespace viennacore;

template <typename NumericType, int D>
class DesorptionSource : public viennaray::Source<NumericType> {
public:
  DesorptionSource(const std::vector<Vec3D<NumericType>> &points,
                   const std::vector<Vec3D<NumericType>> &normals,
                   const std::vector<NumericType> &desorptionRates,
                   const int numRaysPerPoint)
      : points_(points), normals_(normals), desorptionRates_(desorptionRates),
        numRaysPerPoint_(numRaysPerPoint) {}

  std::array<Vec3D<NumericType>, 2>
  getOriginAndDirection(const size_t idx, RNG &RngState) const override {
    size_t pointIdx = idx / numRaysPerPoint_;
    auto direction = viennaray::ReflectionDiffuse<NumericType, D>(
        normals_[pointIdx], RngState);
    return {points_[pointIdx], direction};
  }

  [[nodiscard]] size_t getNumPoints() const override {
    return points_.size() * numRaysPerPoint_;
  }

  NumericType getInitialRayWeight(const size_t idx) const override {
    return desorptionRates_[idx / numRaysPerPoint_] / numRaysPerPoint_;
  }

  NumericType getSourceArea() const override { return 1.; }

private:
  const std::vector<Vec3D<NumericType>> &points_;
  const std::vector<Vec3D<NumericType>> &normals_;
  const std::vector<NumericType> &desorptionRates_;
  const int numRaysPerPoint_;
};

template <typename NumericType, int D> class AtomicLayerProcess {
  using TranslatorType = std::unordered_map<unsigned long, unsigned long>;
  using DomainType = SmartPointer<Domain<NumericType, D>>;

public:
  AtomicLayerProcess() = default;
  explicit AtomicLayerProcess(DomainType domain) : pDomain_(domain) {}
  // Constructor for a process with a pre-configured process model.
  AtomicLayerProcess(
      DomainType domain,
      SmartPointer<ProcessModelCPU<NumericType, D>> passedProcessModel)
      : pDomain_(domain), pModel_(passedProcessModel) {}

  // Set the process model. This can be either a pre-configured process model or
  // a custom process model. A custom process model must interface the
  // ProcessModel class.
  void setProcessModel(
      SmartPointer<ProcessModelCPU<NumericType, D>> passedProcessModel) {
    pModel_ = passedProcessModel;
  }

  // Set the process domain.
  void setDomain(DomainType domain) { pDomain_ = domain; }

  // Set the source direction, where the rays should be traced from.
  void setSourceDirection(viennaray::TraceDirection passedDirection) {
    rayTracingParams_.sourceDirection = passedDirection;
  }

  void setDesorptionRates(std::vector<NumericType> desorptionRates) {
    desorptionRates_ = std::move(desorptionRates);
  }

  void setPulseTime(NumericType pulseTime) { pulseTime_ = pulseTime; }

  void setCoverageTimeStep(NumericType coverageTimeStep) {
    coverageTimeStep_ = coverageTimeStep;
  }

  void setNumCycles(unsigned int numCycles) { numCycles_ = numCycles; }

  // Specify the number of rays to be traced for each particle throughout the
  // process. The total count of rays is the product of this number and the
  // number of points in the process geometry.
  void setNumberOfRaysPerPoint(unsigned numRays) {
    rayTracingParams_.raysPerPoint = numRays;
  }

  // Set the integration scheme for solving the level-set equation.
  // Possible integration schemes are specified in
  // viennals::IntegrationSchemeEnum.
  void setIntegrationScheme(
      viennals::IntegrationSchemeEnum passedIntegrationScheme) {
    advectionParams_.integrationScheme = passedIntegrationScheme;
  }

  // Enable the use of random seeds for ray tracing. This is useful to
  // prevent the formation of artifacts in the flux calculation.
  void enableRandomSeeds() { rayTracingParams_.useRandomSeeds = true; }

  // Disable the use of random seeds for ray tracing.
  void disableRandomSeeds() { rayTracingParams_.useRandomSeeds = false; }

  // Run the process.
  void apply() {

    checkInput();

    pModel_->initialize(pDomain_, 0.);
    auto name = pModel_->getProcessName().value_or("default");
    if (static_cast<int>(pDomain_->getMetaDataLevel()) > 1) {
      pDomain_->clearMetaData(false); // clear previous metadata (without domain
      // metadata)
      pDomain_->addMetaData(pModel_->getProcessMetaData());
    }

    // ---------- Process Setup ---------
    Timer processTimer;
    processTimer.start();

    auto surfaceModel = pModel_->getSurfaceModel();
    auto velocityField = pModel_->getVelocityField();
    const NumericType gridDelta = pDomain_->getGrid().getGridDelta();
    auto const logLevel = Logger::getLogLevel();

    auto diskMesh = viennals::Mesh<NumericType>::New();
    auto translator = SmartPointer<TranslatorType>::New();
    viennals::ToDiskMesh<NumericType, D> meshConverter(diskMesh);
    meshConverter.setTranslator(translator);
    if (pDomain_->getMaterialMap()) {
      meshConverter.setMaterialMap(
          pDomain_->getMaterialMap()->getMaterialMap());
    }

    auto transField = SmartPointer<TranslationField<NumericType, D>>::New(
        pModel_->getVelocityField(), pDomain_->getMaterialMap());
    transField->setTranslator(translator);

    viennals::Advect<NumericType, D> advectionKernel;
    advectionKernel.setVelocityField(transField);
    advectionKernel.setIntegrationScheme(advectionParams_.integrationScheme);
    advectionKernel.setTimeStepRatio(advectionParams_.timeStepRatio);
    advectionKernel.setSaveAdvectionVelocities(advectionParams_.velocityOutput);
    advectionKernel.setDissipationAlpha(advectionParams_.dissipationAlpha);
    advectionKernel.setIgnoreVoids(advectionParams_.ignoreVoids);
    advectionKernel.setCheckDissipation(advectionParams_.checkDissipation);
    advectionKernel.setAdvectionTime(1.);

    for (auto dom : pDomain_->getLevelSets()) {
      meshConverter.insertNextLevelSet(dom);
      advectionKernel.insertNextLevelSet(dom);
    }

    // --------- Setup for ray tracing -----------

    viennaray::Trace<NumericType, D> rayTracer;

    // Map the domain boundary to the ray tracing boundaries
    viennaray::BoundaryCondition rayBoundaryCondition[D];
    if (rayTracingParams_.ignoreFluxBoundaries) {
      for (unsigned i = 0; i < D; ++i)
        rayBoundaryCondition[i] = viennaray::BoundaryCondition::IGNORE;
    } else {
      for (unsigned i = 0; i < D; ++i)
        rayBoundaryCondition[i] = util::convertBoundaryCondition(
            pDomain_->getGrid().getBoundaryConditions(i));
    }
    rayTracer.setBoundaryConditions(rayBoundaryCondition);
    rayTracer.setSourceDirection(rayTracingParams_.sourceDirection);
    rayTracer.setNumberOfRaysPerPoint(rayTracingParams_.raysPerPoint);
    rayTracer.setUseRandomSeeds(rayTracingParams_.useRandomSeeds);
    rayTracer.setCalculateFlux(false);

    if (auto source = pModel_->getSource()) {
      rayTracer.setSource(source);
      Logger::getInstance().addInfo("Using custom source.").print();
    }
    if (auto primaryDirection = pModel_->getPrimaryDirection()) {
      Logger::getInstance()
          .addInfo("Using primary direction: " +
                   util::arrayToString(primaryDirection.value()))
          .print();
      rayTracer.setPrimaryDirection(primaryDirection.value());
    }

    if (logLevel >= 5) {
      // debug output
      std::stringstream ss;
      ss << "Atomic Layer Process: " << name << "\n"
         << "Number of cycles: " << numCycles_ << "\n"
         << "Pulse time: " << pulseTime_ << "\n"
         << "Purge pulse time: " << purgePulseTime_ << "\n"
         << "Coverage time step: " << coverageTimeStep_ << "\n"
         << "Number of particles: " << pModel_->getParticleTypes().size()
         << "\n"
         << "Grid Delta: " << gridDelta << "\n"
         << "Advection Parameters: " << advectionParams_.toMetaDataString()
         << "\n"
         << "Ray Tracing Parameters: " << rayTracingParams_.toMetaDataString()
         << "\n";
      Logger::getInstance().addDebug(ss.str()).print();
    }

    // Determine whether there are process parameters used in ray tracing
    surfaceModel->initializeProcessParameters();
    const bool useProcessParams =
        surfaceModel->getProcessParameters() != nullptr;
    if (useProcessParams)
      Logger::getInstance().addInfo("Using process parameters.").print();

    size_t counter = 0;
    int numCycles = 0;
    while (numCycles++ < numCycles_) {
      Logger::getInstance()
          .addInfo("Cycle: " + std::to_string(numCycles) + "/" +
                   std::to_string(numCycles_))
          .print();

      advectionKernel.prepareLS();

      meshConverter.apply();
      auto numPoints = diskMesh->nodes.size();
      surfaceModel->initializeCoverages(numPoints);

      auto const materialIds =
          *diskMesh->getCellData().getScalarData("MaterialIds");
      auto const points = diskMesh->getNodes();
      auto normals = *diskMesh->getCellData().getVectorData("Normals");
      rayTracer.setGeometry(points, normals, gridDelta);
      rayTracer.setMaterialIds(materialIds);

      auto fluxes = viennals::PointData<NumericType>::New();
      NumericType time = 0.;
      int pulseCounter = 0;

      while (time < pulseTime_) {
#ifdef VIENNATOOLS_PYTHON_BUILD
        if (PyErr_CheckSignals() != 0)
          throw pybind11::error_already_set();
#endif

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

        fluxes->clear();
        std::size_t particleIdx = 0;
        for (auto &particle : pModel_->getParticleTypes()) {
          rayTracer.setParticleType(particle);
          rayTracer.apply();

          // fill up fluxes vector with fluxes from this particle type
          auto numFluxes = particle->getLocalDataLabels().size();
          auto &localData = rayTracer.getLocalData();
          for (int i = 0; i < numFluxes; ++i) {
            auto flux = std::move(localData.getVectorData(i));

            // normalize fluxes
            rayTracer.normalizeFlux(flux);
            rayTracer.smoothFlux(flux, rayTracingParams_.smoothingNeighbors);
            fluxes->insertNextScalarData(std::move(flux),
                                         localData.getVectorDataLabel(i));
          }

          ++particleIdx;
        }

        // move coverages back to model
        moveRayDataToPointData(surfaceModel->getCoverages(), rayTraceCoverages);
        surfaceModel->updateCoverages(fluxes, materialIds);

        // print intermediate output
        if (logLevel >= 3) {
          for (size_t idx = 0; idx < fluxes->getScalarDataSize(); idx++) {
            auto label = fluxes->getScalarDataLabel(idx);
            diskMesh->getCellData().insertNextScalarData(
                *fluxes->getScalarData(idx), label);
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

        if (logLevel >= 2) {
          std::stringstream stream;
          stream << std::fixed << std::setprecision(4) << "Pulse time: " << time
                 << " / " << pulseTime_ << " " << units::Time::toShortString();
          Logger::getInstance().addInfo(stream.str()).print();
        }
      } // end of gas pulse

      if (purgePulseTime_ > 0.) {
        Logger::getInstance().addInfo("Purge pulse ...").print();
        if (desorptionRates_.size() != pModel_->getParticleTypes().size()) {
          Logger::getInstance()
              .addError("Desorption rates not set for all particle types.")
              .print();
        }

        auto purgeFluxes = viennals::PointData<NumericType>::New();

        viennaray::Trace<NumericType, D> purgeTracer;
        purgeTracer.setSourceDirection(rayTracingParams_.sourceDirection);
        purgeTracer.setBoundaryConditions(rayBoundaryCondition);
        purgeTracer.setUseRandomSeeds(rayTracingParams_.useRandomSeeds);
        purgeTracer.setCalculateFlux(false);
        purgeTracer.setGeometry(points, normals, gridDelta);
        purgeTracer.setMaterialIds(materialIds);

        // move coverages to ray tracer
        auto rayTraceCoverages =
            movePointDataToRayData(surfaceModel->getCoverages());
        purgeTracer.setGlobalData(rayTraceCoverages);

        std::size_t particleIdx = 0;
        for (auto &particle : pModel_->getParticleTypes()) {
          auto desorb = rayTraceCoverages.getVectorData(particleIdx);
          for (auto &c : desorb)
            c = c * desorptionRates_[particleIdx] * purgePulseTime_;
          auto source = std::make_shared<DesorptionSource<NumericType, D>>(
              points, normals, desorb, 100);
          purgeTracer.setSource(source);

          purgeTracer.setParticleType(particle);
          purgeTracer.apply();

          auto const numFluxes = particle->getLocalDataLabels().size();
          auto &localData = purgeTracer.getLocalData();
          for (int i = 0; i < numFluxes; ++i) {
            auto flux = std::move(localData.getVectorData(i));
            purgeTracer.smoothFlux(flux, rayTracingParams_.smoothingNeighbors);
            purgeFluxes->insertNextScalarData(std::move(flux),
                                              localData.getVectorDataLabel(i));
          }

          ++particleIdx;
        }

        surfaceModel->updateCoverages(purgeFluxes, materialIds);

      } // end of purge pulse

      // calculate velocities
      auto velocities =
          surfaceModel->calculateVelocities(fluxes, points, materialIds);
      pModel_->getVelocityField()->prepare(pDomain_, velocities, 0.);
      if (pModel_->getVelocityField()->getTranslationFieldOptions() == 2)
        transField->buildKdTree(points);

      // print intermediate output
      if (logLevel >= 3) {
        diskMesh->getCellData().insertNextScalarData(*velocities, "velocities");
        printDiskMesh(diskMesh, name + "_" + std::to_string(counter) + ".vtp");
        counter++;
      }

      // move surface
      advectionKernel.apply();
    }

    processTimer.finish();

    Logger::getInstance().addTiming("\nProcess " + name, processTimer).print();
  }

private:
  void printDiskMesh(SmartPointer<viennals::Mesh<NumericType>> mesh,
                     std::string name) const {
    viennals::VTKWriter<NumericType> writer(mesh, std::move(name));
    writer.setMetaData(pDomain_->getMetaData());
    writer.apply();
  }

  viennaray::TracingData<NumericType> movePointDataToRayData(
      SmartPointer<viennals::PointData<NumericType>> pointData) {
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
      viennaray::TracingData<NumericType> &rayData) {
    pointData->clear();
    const auto numData = rayData.getVectorData().size();
    for (size_t i = 0; i < numData; ++i)
      pointData->insertNextScalarData(std::move(rayData.getVectorData(i)),
                                      rayData.getVectorDataLabel(i));
  }

  void checkInput() const {
    if (!pDomain_) {
      Logger::getInstance()
          .addError("No domain passed to AtomicLayerProcess.")
          .print();
    }

    if (pDomain_->getLevelSets().size() == 0) {
      Logger::getInstance()
          .addError("No level sets in domain passed to AtomicLayerProcess.")
          .print();
    }

    if (!pModel_) {
      Logger::getInstance()
          .addError("No process model passed to AtomicLayerProcess.")
          .print();
    }

    if (!pModel_->getSurfaceModel()) {
      Logger::getInstance()
          .addError("No surface model passed to AtomicLayerProcess.")
          .print();
    }

    if (!pModel_->getVelocityField()) {
      Logger::getInstance()
          .addError("No velocity field passed to AtomicLayerProcess.")
          .print();
    }

    if (pModel_->getParticleTypes().empty()) {
      Logger::getInstance()
          .addError("No particle types specified for ray tracing in "
                    "AtomicLayerProcess.")
          .print();
    }

    if (pModel_->getGeometricModel()) {
      Logger::getInstance()
          .addWarning("Geometric model not supported in AtomicLayerProcess.")
          .print();
    }

    if (pModel_->getAdvectionCallback()) {
      Logger::getInstance()
          .addWarning("Advection callback not supported in AtomicLayerProcess.")
          .print();
    }
  }

  DomainType pDomain_;
  SmartPointer<ProcessModelCPU<NumericType, D>> pModel_;

  AdvectionParameters advectionParams_;
  RayTracingParameters rayTracingParams_;

  unsigned int numCycles_ = 0;
  NumericType pulseTime_ = 0.;
  NumericType purgePulseTime_ = 0.;
  NumericType coverageTimeStep_ = 1.;
  std::vector<NumericType> desorptionRates_;
};

PS_PRECOMPILE_PRECISION_DIMENSION(AtomicLayerProcess)

} // namespace viennaps
