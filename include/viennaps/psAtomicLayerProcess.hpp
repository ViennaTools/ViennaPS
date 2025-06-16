#pragma once

#include "psDomain.hpp"
#include "psProcessModel.hpp"
#include "psTranslationField.hpp"
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
  using translatorType = std::unordered_map<unsigned long, unsigned long>;
  using psDomainType = SmartPointer<Domain<NumericType, D>>;

public:
  AtomicLayerProcess() = default;
  explicit AtomicLayerProcess(psDomainType domain) : pDomain_(domain) {}
  // Constructor for a process with a pre-configured process model.
  AtomicLayerProcess(
      psDomainType domain,
      SmartPointer<ProcessModel<NumericType, D>> passedProcessModel)
      : pDomain_(domain), pModel_(passedProcessModel) {}

  // Set the process model. This can be either a pre-configured process model or
  // a custom process model. A custom process model must interface the
  // ProcessModel class.
  void setProcessModel(
      SmartPointer<ProcessModel<NumericType, D>> passedProcessModel) {
    pModel_ = passedProcessModel;
  }

  // Set the process domain.
  void setDomain(psDomainType domain) { pDomain_ = domain; }

  // Set the source direction, where the rays should be traced from.
  void setSourceDirection(viennaray::TraceDirection passedDirection) {
    sourceDirection_ = passedDirection;
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
  void setNumberOfRaysPerPoint(unsigned numRays) { raysPerPoint_ = numRays; }

  // Set the integration scheme for solving the level-set equation.
  // Possible integration schemes are specified in
  // viennals::IntegrationSchemeEnum.
  void setIntegrationScheme(
      viennals::IntegrationSchemeEnum passedIntegrationScheme) {
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
    Timer processTimer;
    processTimer.start();

    auto name = pModel_->getProcessName().value_or("default");

    const NumericType gridDelta = pDomain_->getGrid().getGridDelta();
    auto diskMesh = viennals::Mesh<NumericType>::New();
    auto translator = SmartPointer<translatorType>::New();
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
    advectionKernel.setIntegrationScheme(integrationScheme_);
    advectionKernel.setAdvectionTime(1.);

    for (auto dom : pDomain_->getLevelSets()) {
      meshConverter.insertNextLevelSet(dom);
      advectionKernel.insertNextLevelSet(dom);
    }

    /* --------- Setup for ray tracing ----------- */

    viennaray::BoundaryCondition rayBoundaryCondition[D];
    viennaray::Trace<NumericType, D> rayTracer;

    // Map the domain boundary to the ray tracing boundaries
    for (unsigned i = 0; i < D; ++i)
      rayBoundaryCondition[i] = util::convertBoundaryCondition(
          pDomain_->getGrid().getBoundaryConditions(i));

    rayTracer.setSourceDirection(sourceDirection_);
    rayTracer.setNumberOfRaysPerPoint(raysPerPoint_);
    rayTracer.setBoundaryConditions(rayBoundaryCondition);
    rayTracer.setUseRandomSeeds(useRandomSeeds_);
    auto primaryDirection = pModel_->getPrimaryDirection();
    if (primaryDirection) {
      Logger::getInstance()
          .addInfo("Using primary direction: " +
                   util::arrayToString(primaryDirection.value()))
          .print();
      rayTracer.setPrimaryDirection(primaryDirection.value());
    }
    rayTracer.setCalculateFlux(false);

    // initialize particle data logs
    auto const numParticles = pModel_->getParticleTypes().size();
    particleDataLogs_.resize(numParticles);
    for (std::size_t i = 0; i < numParticles; i++) {
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
      Logger::getInstance().addInfo("Using process parameters.").print();

    size_t counter = 0;
    int numCycles = 0;
    while (numCycles++ < numCycles_) {
      Logger::getInstance()
          .addInfo("Cycle: " + std::to_string(numCycles) + "/" +
                   std::to_string(numCycles_))
          .print();

      meshConverter.apply();
      auto numPoints = diskMesh->nodes.size();
      surfaceModel->initializeCoverages(numPoints);
      auto rates = viennals::PointData<NumericType>::New();
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

        Logger::getInstance()
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
        for (auto &particle : pModel_->getParticleTypes()) {
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
        if (Logger::getLogLevel() >= 4) {
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
      } // end of gas pulse

      if (purgePulseTime_ > 0.) {
        Logger::getInstance().addInfo("Purge pulse ...").print();
        if (desorptionRates_.size() != numParticles) {
          Logger::getInstance()
              .addError("Desorption rates not set for all particle types.")
              .print();
        }

        auto purgeRates = viennals::PointData<NumericType>::New();

        viennaray::Trace<NumericType, D> purgeTracer;
        purgeTracer.setSourceDirection(sourceDirection_);
        purgeTracer.setBoundaryConditions(rayBoundaryCondition);
        purgeTracer.setUseRandomSeeds(useRandomSeeds_);
        purgeTracer.setCalculateFlux(false);
        purgeTracer.setGeometry(points, normals, gridDelta);
        purgeTracer.setMaterialIds(materialIds);
        if (primaryDirection)
          purgeTracer.setPrimaryDirection(primaryDirection.value());

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

          // fill up rates vector with rates from this particle type
          auto const numRates = particle->getLocalDataLabels().size();
          auto &localData = purgeTracer.getLocalData();
          for (int i = 0; i < numRates; ++i) {
            auto rate = std::move(localData.getVectorData(i));
            purgeTracer.smoothFlux(rate);
            purgeRates->insertNextScalarData(std::move(rate),
                                             localData.getVectorDataLabel(i));
          }

          ++particleIdx;
        }

        surfaceModel->updateCoverages(purgeRates, materialIds);

      } // end of purge pulse

      // get velocities
      auto velocities =
          surfaceModel->calculateVelocities(rates, points, materialIds);
      pModel_->getVelocityField()->prepare(pDomain_, velocities, 0.);
      if (pModel_->getVelocityField()->getTranslationFieldOptions() == 2)
        transField->buildKdTree(points);

      // print debug output
      if (Logger::getLogLevel() >= 4) {
        diskMesh->getCellData().insertNextScalarData(*velocities, "velocities");
        printDiskMesh(diskMesh, name + "_" + std::to_string(counter) + ".vtp");
        counter++;
      }

      advectionKernel.apply();
    }

    processTimer.finish();

    Logger::getInstance().addTiming("\nProcess " + name, processTimer).print();
  }

  void writeParticleDataLogs(const std::string &fileName) {
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
  void printDiskMesh(SmartPointer<viennals::Mesh<NumericType>> mesh,
                     std::string name) const {
    viennals::VTKWriter<NumericType>(mesh, std::move(name)).apply();
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

  psDomainType pDomain_;
  SmartPointer<ProcessModel<NumericType, D>> pModel_;

  viennaray::TraceDirection sourceDirection_ =
      D == 3 ? viennaray::TraceDirection::POS_Z
             : viennaray::TraceDirection::POS_Y;
  viennals::IntegrationSchemeEnum integrationScheme_ =
      viennals::IntegrationSchemeEnum::ENGQUIST_OSHER_1ST_ORDER;
  unsigned raysPerPoint_ = 1000;
  bool useRandomSeeds_ = true;
  std::vector<viennaray::DataLog<NumericType>> particleDataLogs_;

  unsigned int numCycles_ = 0;
  NumericType pulseTime_ = 0.;
  NumericType purgePulseTime_ = 0.;
  NumericType coverageTimeStep_ = 1.;
  std::vector<NumericType> desorptionRates_;
};

} // namespace viennaps
