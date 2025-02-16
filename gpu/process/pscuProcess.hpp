#pragma once

#include <cassert>
#include <cstring>

#include <gpu/vcContext.hpp>

#include <lsAdvect.hpp>
#include <lsDomain.hpp>
#include <lsMesh.hpp>
#include <lsToDiskMesh.hpp>
#include <lsToSurfaceMeshRefined.hpp>

#include <pscuProcessModel.hpp>

#include <psAdvectionCallback.hpp>
#include <psDomain.hpp>
#include <psTranslationField.hpp>
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
  Process(Context context) : context_(context) {}

  Process(Context context, DomainType domain, ModelType model,
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

  // Specify the number of rays to be traced for each particle throughout the
  // process. The total count of rays is the product of this number and the
  // number of points in the process geometry.
  void setNumberOfRaysPerPoint(long raysPerPoint) {
    raysPerPoint_ = raysPerPoint;
  }

  // Set the number of iterations to initialize the coverages.
  void setMaxCoverageInitIterations(long iterations) {
    coverateIterations_ = iterations;
  }

  void setPeriodicBoundary(const bool periodic) {
    periodicBoundary_ = periodic;
  }

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

  void setSmoothingRadius(NumericType pSmoothFlux) {
    smoothFlux_ = pSmoothFlux;
  }

  // Set the CFL (Courant-Friedrichs-Levy) condition to use during surface
  // advection in the level-set. The CFL condition defines the maximum distance
  // a surface is allowed to move in a single advection step. It MUST be below
  // 0.5 to guarantee numerical stability. Defaults to 0.4999.
  void setTimeStepRatio(NumericType cfl) { timeStepRatio_ = cfl; }

  // Sets the minimum time between printing intermediate results during the
  // process. If this is set to a non-positive value, no intermediate results
  // are printed.
  void setPrintTimeInterval(NumericType passedTime) { printTime_ = passedTime; }

  template <class ParamType> void setProcessParams(const ParamType &params) {
    rayTrace_.setParameters(params);
  }

  SmartPointer<viennals::Mesh<NumericType>> calculateFlux() {

    const auto name = model_->getProcessName().value_or("default");
    const NumericType gridDelta = domain_->getGrid().getGridDelta();

    auto diskMesh = SmartPointer<viennals::Mesh<NumericType>>::New();
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
    Timer transTimer;
    auto surfMesh = SmartPointer<viennals::Mesh<NumericType>>::New();
    auto elementKdTree =
        SmartPointer<KDTree<NumericType, std::array<NumericType, 3>>>::New();
    CreateSurfaceMesh<NumericType, NumericType, D> surfMeshConverter(
        domain_->getLevelSets().back(), surfMesh, elementKdTree);

    /* --------- Setup for ray tracing ----------- */
    unsigned int numRates = 0;
    IndexMap fluxesIndexMap;
    Timer rtTimer;

    if (!rayTracerInitialized_) {
      rayTrace_.setPipeline(model_->getPipelineFileName(),
                            context_->modulePath);
      rayTrace_.setNumberOfRaysPerPoint(raysPerPoint_);
      rayTrace_.setUseRandomSeeds(useRandomSeeds_);
      rayTrace_.setPeriodicBoundary(periodicBoundary_);
      for (auto &particle : model_->getParticleTypes()) {
        rayTrace_.insertNextParticle(particle);
      }
      numRates = rayTrace_.prepareParticlePrograms();
      fluxesIndexMap = IndexMap(rayTrace_.getParticles());
    }
    rayTracerInitialized_ = true;

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

    TriangleMesh mesh(gridDelta, surfMesh);
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
    ElementToPointData<NumericType>(rayTrace_.getResults(), fluxes,
                                    rayTrace_.getParticles(), elementKdTree,
                                    diskMesh, surfMesh, gridDelta)
        .apply();

    mergeScalarData(diskMesh->getCellData(), fluxes);

    return diskMesh;
  }

  void apply() {
    if (checkInput())
      return;

    /* ---------- Process Setup --------- */
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
    const NumericType gridDelta = domain_->getGrid().getGridDelta();

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
    advectionKernel.setIntegrationScheme(integrationScheme_);
    advectionKernel.setTimeStepRatio(timeStepRatio_);

    for (auto ls : domain_->getLevelSets()) {
      diskMeshConverter.insertNextLevelSet(ls);
      advectionKernel.insertNextLevelSet(ls);
    }
    advectionKernel.prepareLS();

    /* --------- Setup triangulated surface mesh ----------- */
    auto surfMesh = SmartPointer<viennals::Mesh<float>>::New();
    auto elementKdTree =
        SmartPointer<KDTree<NumericType, std::array<NumericType, 3>>>::New();
    viennals::ToSurfaceMeshRefined<NumericType, float, D> surfMeshConverter(
        domain_->getLevelSets().back(), surfMesh, elementKdTree);
    // surfMeshConverter.setCheckNodeForDouble(false);

    /* --------- Setup for ray tracing ----------- */
    unsigned int numRates = 0;
    unsigned int numCov = 0;
    IndexMap fluxesIndexMap;

    if (!rayTracerInitialized_) {
      rayTrace_.setPipeline(model_->getPipelineFileName(),
                            context_->modulePath);
      rayTrace_.setNumberOfRaysPerPoint(raysPerPoint_);
      rayTrace_.setUseRandomSeeds(useRandomSeeds_);
      rayTrace_.setPeriodicBoundary(periodicBoundary_);
      for (auto &particle : model_->getParticleTypes()) {
        rayTrace_.insertNextParticle(particle);
      }
      numRates = rayTrace_.prepareParticlePrograms();
      fluxesIndexMap = IndexMap(rayTrace_.getParticles());
    }
    rayTracerInitialized_ = true;

    /* --------- Meshing ----------- */

    meshTimer.start();
    surfMeshTimer.start();
    surfMeshConverter.apply(); // also build elementKdTree
    surfMeshTimer.finish();
    diskMeshConverter.apply();
    assert(diskMesh->nodes.size() > 0);
    assert(surfMesh->nodes.size() > 0);
    transField->buildKdTree(diskMesh->nodes);
    TriangleMesh mesh(gridDelta, surfMesh);
    rayTrace_.setGeometry(mesh);
    meshTimer.finish();
    Logger::getInstance().addTiming("Geometry generation", meshTimer).print();
    Logger::getInstance()
        .addTiming("Surface mesh generation", surfMeshTimer)
        .print();

    /* --------- Initialize coverages ----------- */
    surfaceModel->initializeSurfaceData(diskMesh->nodes.size());
    if (!coveragesInitialized_)
      surfaceModel->initializeCoverages(diskMesh->nodes.size());
    auto coverages = surfaceModel->getCoverages(); // might be null
    const bool useCoverages = coverages != nullptr;

    if (useCoverages) {
      Logger::getInstance().addInfo("Using coverages.").print();

      Timer timer;
      Logger::getInstance().addInfo("Initializing coverages ... ").print();

      numCov = coverages->getScalarDataSize();
      CudaBuffer d_coverages; // device buffer for coverages

      timer.start();
      for (size_t iterations = 1; iterations <= coverateIterations_;
           iterations++) {

        transTimer.start();
        PointToElementData<NumericType>(d_coverages, coverages,
                                        transField->getKdTree(), surfMesh)
            .apply();
        transTimer.finish();

        rayTrace_.setElementData(d_coverages, numCov);
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
        ElementToPointData<NumericType>(rayTrace_.getResults(), fluxes,
                                        rayTrace_.getParticles(), elementKdTree,
                                        diskMesh, surfMesh, gridDelta)
            .apply();
        transTimer.finish();

        // calculate coverages
        const auto &materialIds =
            *diskMesh->getCellData().getScalarData("MaterialIds");
        surfaceModel->updateCoverages(fluxes, materialIds);

        // output
        if (Logger::getLogLevel() >= 3) {
          // assert(numElements == mesh->triangles.size());

          // downloadCoverages(d_coverages, mesh->getCellData(), coverages,
          //                   numElements);

          // rayTrace_.downloadResultsToPointData(mesh->getCellData());
          // viennals::VTKWriter<NumericType>(
          //     mesh,
          //     name + "_covInit_mesh_" + std::to_string(iterations) + ".vtp")
          //     .apply();
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

      timer.finish();
      Logger::getInstance().addTiming("Coverage initialization", timer).print();
    }

    double previousTimeStep = 0.;
    size_t counter = 0;
    while (remainingTime > 0.) {

      meshTimer.start();
      surfMeshConverter.apply();                // build element KD tree
      transField->buildKdTree(diskMesh->nodes); // build point KD tree
      mesh = TriangleMesh(gridDelta, surfMesh);
      rayTrace_.setGeometry(mesh);
      meshTimer.finish();

      Logger::getInstance().addTiming("Geometry generation", meshTimer).print();

      CudaBuffer d_coverages; // device buffer for coverages
      if (useCoverages) {
        transTimer.start();
        PointToElementData<NumericType>(d_coverages, coverages,
                                        transField->getKdTree(), surfMesh)
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
      auto fluxes = SmartPointer<viennals::PointData<NumericType>>::New();
      transTimer.start();
      ElementToPointData<NumericType>(rayTrace_.getResults(), fluxes,
                                      rayTrace_.getParticles(), elementKdTree,
                                      diskMesh, surfMesh, gridDelta)
          .apply();
      transTimer.finish();

      const auto &materialIds =
          *diskMesh->getCellData().getScalarData("MaterialIds");
      auto velocities = surfaceModel->calculateVelocities(
          fluxes, diskMesh->nodes, materialIds);
      model_->getVelocityField()->prepare(domain_, velocities,
                                          processDuration_ - remainingTime);
      assert(velocities->size() == materialIds.size());

      if (Logger::getLogLevel() >= 4) {
        {
          // CudaBuffer dummy;
          // PointToElementData<NumericType>(
          //     dummy, fluxes, transField->getKdTree(), surfMesh, true, false)
          //     .apply();
          rayTrace_.downloadResultsToPointData(surfMesh->getCellData());
          viennals::VTKWriter<NumericType>(
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

      if (Logger::getLogLevel() >= 2) {
        std::stringstream stream;
        stream << std::fixed << std::setprecision(4)
               << "Process time: " << processDuration_ - remainingTime << " / "
               << processDuration_;
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

  // static void
  // downloadCoverages(CudaBuffer &d_coverages,
  //                   viennals::PointData<NumericType> &elementData,
  //                   SmartPointer<viennals::PointData<NumericType>>
  //                   &coverages, unsigned int numElements) {

  //   auto numCov = coverages->getScalarDataSize() + 1; // + material ids
  //   NumericType *temp = new NumericType[numElements * numCov];
  //   d_coverages.download(temp, numElements * numCov);

  //   // material IDs at front
  //   {
  //     auto matIds = elementData.getScalarData("Material");
  //     if (matIds == nullptr) {
  //       std::vector<NumericType> tmp(numElements);
  //       elementData.insertNextScalarData(std::move(tmp), "Material");
  //       matIds = elementData.getScalarData("Material");
  //     }
  //     if (matIds->size() != numElements)
  //       matIds->resize(numElements);
  //     std::memcpy(matIds->data(), temp, numElements * sizeof(NumericType));
  //   }

  //   for (unsigned i = 0; i < numCov - 1; i++) {
  //     auto covName = coverages->getScalarDataLabel(i);
  //     auto cov = elementData.getScalarData(covName);
  //     if (cov == nullptr) {
  //       std::vector<NumericType> covInit(numElements);
  //       elementData.insertNextScalarData(std::move(covInit), covName);
  //       cov = elementData.getScalarData(covName);
  //     }
  //     if (cov->size() != numElements)
  //       cov->resize(numElements);
  //     std::memcpy(cov->data(), temp + (i + 1) * numElements,
  //                 numElements * sizeof(NumericType));
  //   }

  //   delete temp;
  // }

  bool checkInput() {
    if (!model_) {
      Logger::getInstance()
          .addWarning("No process model passed to Process.")
          .print();
      return true;
    }
    const auto name = model_->getProcessName().value_or("default");

    if (!domain_) {
      Logger::getInstance().addWarning("No domain passed to Process.").print();
      return true;
    }

    if (domain_->getLevelSets().empty()) {
      Logger::getInstance().addWarning("No level sets in domain.").print();
      return true;
    }

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

  Context context_;
  Trace<NumericType, D> rayTrace_ = Trace<NumericType, D>(context_);

  DomainType domain_;
  ModelType model_;

  NumericType processDuration_;
  unsigned raysPerPoint_ = 1000;
  unsigned coverateIterations_ = 20;
  viennals::IntegrationSchemeEnum integrationScheme_ =
      viennals::IntegrationSchemeEnum::ENGQUIST_OSHER_1ST_ORDER;

  NumericType smoothFlux_ = 1.;
  NumericType processTime_ = 0;
  NumericType printTime_ = 0.;
  NumericType timeStepRatio_ = 0.4999;

  bool useRandomSeeds_ = true;
  bool periodicBoundary_ = false;
  bool coveragesInitialized_ = false;
  bool rayTracerInitialized_ = false;
};

} // namespace gpu
} // namespace viennaps