#pragma once

#include <cassert>
#include <cstring>

#include <gpu/vcContext.hpp>

#include <lsAdvect.hpp>
#include <lsDomain.hpp>
#include <lsMesh.hpp>
#include <lsToDiskMesh.hpp>

#include <pscuProcessModel.hpp>

#include <psAdvectionCallback.hpp>
#include <psDomain.hpp>
#include <psProcessParams.hpp>
#include <psTranslationField.hpp>
#include <psUnits.hpp>
#include <psUtils.hpp>
#include <psVelocityField.hpp>

#include <curtIndexMap.hpp>
#include <curtParticle.hpp>
#include <curtSmoothing.hpp>
#include <curtTrace.hpp>

#include <utElementToPointData.hpp>
#include <utPointToElementData.hpp>

namespace viennaps {

namespace gpu {

using namespace viennacore;

template <typename NumericType, int D> class Process {
  using DomainType = SmartPointer<::viennaps::Domain<NumericType, D>>;
  using ModelType = SmartPointer<ProcessModel<NumericType, D>>;
  using TranslatorType = std::unordered_map<unsigned long, unsigned long>;

public:
  Process(Context &context) : context_(context) {}

  Process(Context &context, DomainType domain, ModelType model,
          NumericType duration = 0.0)
      : context_(context), domain_(domain), model_(model),
        processDuration_(duration) {}

  // Set the process model. This can be either a pre-configured process model or
  // a custom process model. A custom process model must interface the
  // pscuProcessModel class.
  void setProcessModel(ModelType processModel) {
    model_ = processModel;
    rayTracerInitialized_ = false;
  }

  // Set the process domain.
  void setDomain(DomainType domain) { domain_ = domain; }

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
  void setMaxCoverageInitIterations(unsigned maxIt) {
    coverateIterations_ = maxIt;
  }

  // Specify the number of rays to be traced for each particle throughout the
  // process. The total count of rays is the product of this number and the
  // number of points in the process geometry.
  void setNumberOfRaysPerPoint(unsigned numRays) {
    rayTracingParams_.raysPerPoint = numRays;
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
  void setIntegrationScheme(
      viennals::IntegrationSchemeEnum passedIntegrationScheme) {
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

  SmartPointer<viennals::Mesh<NumericType>> calculateFlux() {

    auto diskMesh = SmartPointer<viennals::Mesh<NumericType>>::New();
    if (checkInput())
      return diskMesh;

    const auto name = model_->getProcessName().value_or("default");
    const NumericType gridDelta = domain_->getGridDelta();
    viennals::ToDiskMesh<NumericType, D> diskMeshConverter(diskMesh);
    if (domain_->getMaterialMap() &&
        domain_->getMaterialMap()->size() == domain_->getLevelSets().size()) {
      diskMeshConverter.setMaterialMap(
          domain_->getMaterialMap()->getMaterialMap());
    }

    for (auto ls : domain_->getLevelSets()) {
      diskMeshConverter.insertNextLevelSet(ls);
    }

    /* --------- Setup triangulated surface mesh ----------- */
    auto surfMesh = SmartPointer<viennals::Mesh<float>>::New();
    auto elementKdTree =
        SmartPointer<KDTree<NumericType, std::array<NumericType, 3>>>::New();
    CreateSurfaceMesh<NumericType, float, D> surfMeshConverter(
        domain_->getLevelSets().back(), surfMesh, elementKdTree);

    /* --------- Setup for ray tracing ----------- */
    unsigned int numRates = 0;
    IndexMap fluxesIndexMap;
    Timer rtTimer;

    if (!rayTracerInitialized_) {
      rayTrace_.setPipeline(model_->getPipelineFileName(), context_.modulePath);
      rayTrace_.setNumberOfRaysPerPoint(rayTracingParams_.raysPerPoint);
      rayTrace_.setUseRandomSeeds(rayTracingParams_.useRandomSeeds);
      rayTrace_.setPeriodicBoundary(periodicBoundary_);
      for (auto &particle : model_->getParticleTypes()) {
        rayTrace_.insertNextParticle(particle);
      }
      numRates = rayTrace_.prepareParticlePrograms();
      fluxesIndexMap = IndexMap(rayTrace_.getParticles());
    }
    rayTracerInitialized_ = true;
    rayTrace_.setParameters(model_->getProcessDataDPtr());

    /* --------- Meshing ----------- */
    Timer meshTimer;
    Timer surfMeshTimer;
    meshTimer.start();
    surfMeshTimer.start();
    surfMeshConverter.apply(); // also build elementKdTree
    surfMeshTimer.finish();
    diskMeshConverter.apply();
    assert(diskMesh->nodes.size() > 0);
    assert(surfMesh->nodes.size() > 0);

    TriangleMesh<float> mesh(gridDelta, surfMesh);
    rayTrace_.setGeometry(mesh);
    meshTimer.finish();
    Logger::getInstance().addTiming("Geometry generation", meshTimer).print();
    Logger::getInstance()
        .addTiming("Surface mesh generation", surfMeshTimer)
        .print();

    // run the ray tracer
    rtTimer.start();
    rayTrace_.apply();
    rtTimer.finish();

    Logger::getInstance()
        .addTiming("Top-down flux calculation", rtTimer)
        .print();

    // extract fluxes on points
    auto fluxes = SmartPointer<viennals::PointData<NumericType>>::New();
    ElementToPointData<NumericType, float>(
        rayTrace_.getResults(), fluxes, rayTrace_.getParticles(), elementKdTree,
        diskMesh, surfMesh, gridDelta * rayTracingParams_.smoothingNeighbors)
        .apply();

    mergeScalarData(diskMesh->getCellData(), fluxes);

    return diskMesh;
  }

  void apply() {
    if (checkInput())
      return;

    /* ---------- Process Setup --------- */

    const unsigned int logLevel = Logger::getLogLevel();
    Timer processTimer;
    Timer rtTimer;
    Timer advTimer;
    Timer transTimer;
    Timer meshTimer;
    Timer surfMeshTimer;
    processTimer.start();

    const auto name = model_->getProcessName().value_or("default");
    auto surfaceModel = model_->getSurfaceModel();
    double remainingTime = processDuration_;
    const NumericType gridDelta = domain_->getGridDelta();

    auto diskMesh = SmartPointer<viennals::Mesh<NumericType>>::New();
    auto translator = SmartPointer<TranslatorType>::New();
    viennals::ToDiskMesh<NumericType, D> diskMeshConverter(diskMesh);
    diskMeshConverter.setTranslator(translator);
    if (domain_->getMaterialMap() &&
        domain_->getMaterialMap()->size() == domain_->getLevelSets().size()) {
      diskMeshConverter.setMaterialMap(
          domain_->getMaterialMap()->getMaterialMap());
    }

    auto transField = SmartPointer<TranslationField<NumericType, D>>::New(
        model_->getVelocityField(), domain_->getMaterialMap());
    transField->setTranslator(translator);

    /* --------- Setup advection kernel ----------- */
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

    for (auto ls : domain_->getLevelSets()) {
      diskMeshConverter.insertNextLevelSet(ls);
      advectionKernel.insertNextLevelSet(ls);
    }
    advectionKernel.prepareLS();

    /* --------- Setup triangulated surface mesh ----------- */
    auto surfMesh = SmartPointer<viennals::Mesh<float>>::New();
    auto elementKdTree =
        SmartPointer<KDTree<NumericType, std::array<NumericType, 3>>>::New();
    CreateSurfaceMesh<NumericType, float, D> surfMeshConverter(
        domain_->getLevelSets().back(), surfMesh, elementKdTree);
    // surfMeshConverter.setCheckNodeForDouble(false);

    /* --------- Setup for ray tracing ----------- */
    unsigned int numRates = 0;
    unsigned int numCov = 0;
    IndexMap fluxesIndexMap;

    if (!rayTracerInitialized_) {
      // Check for periodic boundary conditions
      if (rayTracingParams_.ignoreFluxBoundaries) {
        Logger::getInstance()
            .addWarning("Ignoring flux boundaries not implemented yet.")
            .print();
      } else {
        for (unsigned i = 0; i < D; ++i) {
          if (domain_->getGrid().getBoundaryConditions(i) ==
              viennals::BoundaryConditionEnum::PERIODIC_BOUNDARY) {
            periodicBoundary_ = true;
            break;
          }
        }
      }
      rayTrace_.setPipeline(model_->getPipelineFileName(), context_.modulePath);
      rayTrace_.setNumberOfRaysPerPoint(rayTracingParams_.raysPerPoint);
      rayTrace_.setUseRandomSeeds(rayTracingParams_.useRandomSeeds);
      rayTrace_.setPeriodicBoundary(periodicBoundary_);
      for (auto &particle : model_->getParticleTypes()) {
        rayTrace_.insertNextParticle(particle);
      }
      numRates = rayTrace_.prepareParticlePrograms();
      fluxesIndexMap = IndexMap(rayTrace_.getParticles());
    }
    rayTracerInitialized_ = true;
    rayTrace_.setParameters(model_->getProcessDataDPtr());

    // Determine whether there are process parameters used in ray tracing
    model_->getSurfaceModel()->initializeProcessParameters();
    const bool useProcessParams =
        model_->getSurfaceModel()->getProcessParameters() != nullptr;

    if (useProcessParams)
      Logger::getInstance().addInfo("Using process parameters.").print();

    /* --------- Meshing ----------- */

    meshTimer.start();
    surfMeshTimer.start();
    surfMeshConverter.apply(); // also build elementKdTree
    surfMeshTimer.finish();
    diskMeshConverter.apply();
    assert(diskMesh->nodes.size() > 0);
    assert(surfMesh->nodes.size() > 0);
    transField->buildKdTree(diskMesh->nodes);
    TriangleMesh<float> mesh(gridDelta, surfMesh);
    rayTrace_.setGeometry(mesh);
    meshTimer.finish();
    Logger::getInstance().addTiming("Geometry generation", meshTimer).print();
    Logger::getInstance()
        .addTiming("Surface mesh generation", surfMeshTimer)
        .print();

    /* --------- Initialize coverages ----------- */
    auto numPoints = diskMesh->nodes.size();
    surfaceModel->initializeSurfaceData(numPoints);
    if (!coveragesInitialized_)
      surfaceModel->initializeCoverages(numPoints);
    auto coverages = surfaceModel->getCoverages(); // might be null
    const bool useCoverages = coverages != nullptr;

    std::ofstream covMetricFile;
    if (useCoverages) {
      Timer timer;
      numCov = coverages->getScalarDataSize();
      Logger::getInstance().addInfo("Using coverages.").print();
      // debug output
      if (logLevel >= 5)
        covMetricFile.open(name + "_covMetric.txt");

      if (!coveragesInitialized_) {
        timer.start();
        Logger::getInstance().addInfo("Initializing coverages ... ").print();

        CudaBuffer d_coverages; // device buffer for coverages

        for (size_t iterations = 1; iterations <= coverateIterations_;
             iterations++) {
// We need additional signal handling when running the C++ code from
// the Python bindings to allow interrupts in the Python scripts
#ifdef VIENNAPS_PYTHON_BUILD
          if (PyErr_CheckSignals() != 0)
            throw pybind11::error_already_set();
#endif
          // save current coverages to compare with the new ones
          auto prevStepCoverages =
              SmartPointer<viennals::PointData<NumericType>>::New(*coverages);

          // move coverages to the ray tracer
          transTimer.start();
          PointToElementData<NumericType, float>(
              d_coverages, coverages, transField->getKdTree(), surfMesh)
              .apply();
          rayTrace_.setElementData(d_coverages, numCov);
          transTimer.finish();

          // run the ray tracer
          rtTimer.start();
          rayTrace_.apply();
          rtTimer.finish();

          Logger::getInstance()
              .addTiming("Top-down flux calculation", rtTimer)
              .print();

          // extract fluxes on points
          auto fluxes = SmartPointer<viennals::PointData<NumericType>>::New();
          transTimer.start();
          ElementToPointData<NumericType, float>(
              rayTrace_.getResults(), fluxes, rayTrace_.getParticles(),
              elementKdTree, diskMesh, surfMesh,
              gridDelta * rayTracingParams_.smoothingNeighbors)
              .apply();
          transTimer.finish();

          // calculate coverages
          const auto &materialIds =
              *diskMesh->getCellData().getScalarData("MaterialIds");
          surfaceModel->updateCoverages(fluxes, materialIds);

          // check for convergence
          if (logLevel >= 5) {
            coverages = model_->getSurfaceModel()->getCoverages();
            auto metric = utils::calculateCoverageDeltaMetric(
                coverages, prevStepCoverages);
            for (auto val : metric) {
              covMetricFile << val << ";";
            }
            covMetricFile << "\n";
          }

          // output
          if (logLevel >= 3) {
            downloadCoverages(d_coverages, surfMesh->getCellData(), coverages,
                              surfMesh->template getElements<3>().size());
            rayTrace_.downloadResultsToPointData(surfMesh->getCellData());
            viennals::VTKWriter<float>(surfMesh,
                                       name + "_covInit_flux_" +
                                           std::to_string(iterations) + ".vtp")
                .apply();

            mergeScalarData(diskMesh->getCellData(), fluxes);
            mergeScalarData(diskMesh->getCellData(), coverages);
            viennals::VTKWriter<NumericType>(
                diskMesh,
                name + "_covInit_" + std::to_string(iterations) + ".vtp")
                .apply();
          }

          Logger::getInstance()
              .addInfo("Iteration: " + std::to_string(iterations))
              .print();
        }
        d_coverages.free();
        coveragesInitialized_ = true;

        timer.finish();
        Logger::getInstance()
            .addTiming("Coverage initialization", timer)
            .print();
      }
    } // end coverage initialization

    double previousTimeStep = 0.;
    size_t counter = 0;
    unsigned lsVelCounter = 0;
    while (remainingTime > 0.) {
// We need additional signal handling when running the C++ code from the
// Python bindings to allow interrupts in the Python scripts
#ifdef VIENNAPS_PYTHON_BUILD
      if (PyErr_CheckSignals() != 0)
        throw pybind11::error_already_set();
#endif
      // Expand LS based on the integration scheme
      advectionKernel.prepareLS();
      diskMeshConverter.apply(); //

      auto fluxes = SmartPointer<viennals::PointData<NumericType>>::New();
      meshTimer.start();
      surfMeshConverter.apply();                // build element KD tree
      transField->buildKdTree(diskMesh->nodes); // build point KD tree
      mesh = TriangleMesh<float>(gridDelta, surfMesh);
      rayTrace_.setGeometry(mesh);
      meshTimer.finish();

      Logger::getInstance().addTiming("Geometry generation", meshTimer).print();

      CudaBuffer d_coverages; // device buffer for coverages
      if (useCoverages) {
        transTimer.start();
        PointToElementData<NumericType, float>(
            d_coverages, coverages, transField->getKdTree(), surfMesh)
            .apply();
        transTimer.finish();

        rayTrace_.setElementData(d_coverages, numCov);
      }

      // run the ray tracer
      rtTimer.start();
      rayTrace_.apply();
      rtTimer.finish();

      Logger::getInstance()
          .addTiming("Top-down flux calculation", rtTimer)
          .print();

      // extract fluxes on points
      transTimer.start();
      ElementToPointData<NumericType, float>(
          rayTrace_.getResults(), fluxes, rayTrace_.getParticles(),
          elementKdTree, diskMesh, surfMesh,
          gridDelta * rayTracingParams_.smoothingNeighbors)
          .apply();
      transTimer.finish();

      // save coverages for comparison
      SmartPointer<viennals::PointData<NumericType>> prevStepCoverages;
      if (useCoverages && logLevel >= 5) {
        coverages = model_->getSurfaceModel()->getCoverages();
        prevStepCoverages =
            SmartPointer<viennals::PointData<NumericType>>::New(*coverages);
      }

      // get velocities from fluxes
      const auto &materialIds =
          *diskMesh->getCellData().getScalarData("MaterialIds");
      auto velocities = surfaceModel->calculateVelocities(
          fluxes, diskMesh->nodes, materialIds);

      // calculate coverage metric
      if (useCoverages && logLevel >= 5) {
        auto metric =
            utils::calculateCoverageDeltaMetric(coverages, prevStepCoverages);
        for (auto val : metric) {
          covMetricFile << val << ";";
        }
        covMetricFile << "\n";
      }

      // prepare velocity field
      model_->getVelocityField()->prepare(domain_, velocities,
                                          processDuration_ - remainingTime);
      if (model_->getVelocityField()->getTranslationFieldOptions() == 2)
        transField->buildKdTree(diskMesh->nodes);

      // print debug output
      if (logLevel >= 4) {
        {
          // CudaBuffer dummy;
          // PointToElementData<NumericType>(
          //     dummy, fluxes, transField->getKdTree(), surfMesh, true,
          //     false) .apply();
          downloadCoverages(d_coverages, surfMesh->getCellData(), coverages,
                            surfMesh->template getElements<3>().size());
          rayTrace_.downloadResultsToPointData(surfMesh->getCellData());
          viennals::VTKWriter<float>(
              surfMesh, name + "_flux_" + std::to_string(counter) + ".vtp")
              .apply();
        }

        if (useCoverages)
          mergeScalarData(diskMesh->getCellData(), coverages);

        mergeScalarData(diskMesh->getCellData(), fluxes);

        if (velocities)
          diskMesh->getCellData().insertReplaceScalarData(*velocities,
                                                          "velocities");

        viennals::VTKWriter<NumericType>(
            diskMesh, name + "_" + std::to_string(counter) + ".vtp")
            .apply();

        counter++;
      }
      d_coverages.free();

      // adjust time step near end
      if (remainingTime - previousTimeStep < 0.) {
        advectionKernel.setAdvectionTime(remainingTime);
      }

      // move coverages to top LS
      if (useCoverages)
        moveCoveragesToTopLS(translator, coverages);

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

      // update surface and move coverages to new surface
      diskMeshConverter.apply();
      if (useCoverages)
        updateCoveragesFromAdvectedSurface(translator, coverages);

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

      if (logLevel >= 2) {
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
    Logger::getInstance()
        .addTiming("Top-down flux calculation total time",
                   rtTimer.totalDuration * 1e-9,
                   processTimer.totalDuration * 1e-9)
        .print();
    Logger::getInstance()
        .addTiming("Mesh generation total time", meshTimer.totalDuration * 1e-9,
                   processTimer.totalDuration * 1e-9)
        .print();
    Logger::getInstance()
        .addTiming("Data conversion total time",
                   transTimer.totalDuration * 1e-9,
                   processTimer.totalDuration * 1e-9)
        .print();
  }

private:
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

  static void
  mergeScalarData(viennals::PointData<NumericType> &scalarData,
                  SmartPointer<viennals::PointData<NumericType>> dataToInsert) {
    int numScalarData = dataToInsert->getScalarDataSize();
    for (int i = 0; i < numScalarData; i++) {
      scalarData.insertReplaceScalarData(*dataToInsert->getScalarData(i),
                                         dataToInsert->getScalarDataLabel(i));
    }
  }

  static void
  downloadCoverages(CudaBuffer &d_coverages,
                    viennals::PointData<float> &elementData,
                    SmartPointer<viennals::PointData<NumericType>> &coverages,
                    unsigned int numElements) {

    auto numCov = coverages->getScalarDataSize(); // + material ids
    float *temp = new float[numElements * numCov];
    d_coverages.download(temp, numElements * numCov);

    for (unsigned i = 0; i < numCov - 1; i++) {
      auto covName = coverages->getScalarDataLabel(i);
      std::vector<float> values(numElements);
      std::memcpy(values.data(), &temp[i * numElements],
                  numElements * sizeof(float));
      elementData.insertReplaceScalarData(values, covName);
    }

    delete temp;
  }

  bool checkInput() {
    if (!model_) {
      Logger::getInstance()
          .addWarning("No process model passed to Process.")
          .print();
      return true;
    }

    if (!domain_) {
      Logger::getInstance().addWarning("No domain passed to Process.").print();
      return true;
    }

    if (domain_->getLevelSets().empty()) {
      Logger::getInstance().addWarning("No level sets in domain.").print();
      return true;
    }

    model_->initialize(domain_, processDuration_);
    const auto name = model_->getProcessName().value_or("default");

    if (!model_->getVelocityField()) {
      Logger::getInstance()
          .addWarning("No velocity field in process model: " + name)
          .print();
      return true;
    }

    if (!model_->getSurfaceModel()) {
      Logger::getInstance()
          .addWarning("No surface model in process model: " + name)
          .print();
      return true;
    }

    if (model_->getParticleTypes().empty()) {
      Logger::getInstance()
          .addWarning("No particle types in process model: " + name)
          .print();
      return true;
    }

    if (model_->getPipelineFileName().empty()) {
      Logger::getInstance()
          .addWarning("No pipeline in process model: " + name)
          .print();
      return true;
    }

    return false;
  }

  Context &context_;
  Trace<NumericType, D> rayTrace_ = Trace<NumericType, D>(context_);

  DomainType domain_;
  ModelType model_;

  NumericType processDuration_;
  unsigned coverateIterations_ = 20;
  NumericType processTime_ = 0;

  bool periodicBoundary_ = false;
  bool coveragesInitialized_ = false;
  bool rayTracerInitialized_ = false;

  AdvectionParameters<NumericType> advectionParams_;
  RayTracingParameters<NumericType, D> rayTracingParams_;
};

} // namespace gpu
} // namespace viennaps