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
#include <psTranslationField.hpp>
#include <psVelocityField.hpp>

#include <curtIndexMap.hpp>
#include <curtParticle.hpp>
#include <curtSmoothing.hpp>
#include <curtTracer.hpp>

#include <culsToSurfaceMesh.hpp>

namespace viennaps {

namespace gpu {

using namespace viennacore;

template <typename NumericType, int D> class Process {
  using DomainType = SmartPointer<::viennaps::Domain<NumericType, D>>;
  using ModelType = SmartPointer<ProcessModel<NumericType>>;

public:
  Process(Context context) : context_(context) {}

  Process(Context context, DomainType domain, ModelType model,
          NumericType duration = 0.0)
      : context_(context), domain_(domain), model_(model),
        processDuration_(duration) {}

  void setProcessModel(ModelType processModel) { model_ = processModel; }

  void setDomain(DomainType domain) { domain_ = domain; }

  void setProcessDuration(double duration) { processDuration_ = duration; }

  void setNumberOfRaysPerPoint(long raysPerPoint) {
    raysPerPoint_ = raysPerPoint;
  }

  void setMaxCoverageInitIterations(long iterations) {
    coverateIterations_ = iterations;
  }

  void setPeriodicBoundary(const bool periodic) {
    periodicBoundary_ = periodic;
  }

  void setSmoothFlux(NumericType pSmoothFlux) { smoothFlux_ = pSmoothFlux; }

  void apply() {
    if (checkInput())
      return;

    /* ---------- Process Setup --------- */
    Timer processTimer;
    processTimer.start();

    auto const name = model_->getProcessName();
    auto surfaceModel = model_->getSurfaceModel();
    double remainingTime = processDuration_;
    const NumericType gridDelta = domain_->getGrid().getGridDelta();

    auto diskMesh = SmartPointer<viennals::Mesh<NumericType>>::New();
    viennals::ToDiskMesh<NumericType, D> diskMeshConv(diskMesh);

    /* --------- Setup advection kernel ----------- */
    viennals::Advect<NumericType, D> advectionKernel;
    advectionKernel.setIntegrationScheme(integrationScheme_);

    auto transField = SmartPointer<TranslationField<NumericType>>::New(
        model_->getVelocityField(), domain_->getMaterialMap());
    advectionKernel.setVelocityField(transField);

    for (auto dom : domain_->getLevelSets()) {
      advectionKernel.insertNextLevelSet(dom);
      diskMeshConv.insertNextLevelSet(dom);
    }

    /* --------- Setup element-point translation ----------- */
    auto mesh = rayTrace_.getSurfaceMesh(); // empty mesh
    auto elementKdTree =
        SmartPointer<KDTree<NumericType, std::array<NumericType, 3>>>::New();
    rayTrace_.setKdTree(elementKdTree);

    diskMeshConv.apply(); // creates disk mesh
    assert(diskMesh->nodes.size());

    transField->buildKdTree(diskMesh->nodes);

    /* --------- Setup for ray tracing ----------- */
    unsigned int numRates = 0;
    unsigned int numCov = 0;
    IndexMap fluxesIndexMap;

    if (!rayTracerInitialized_) {
      rayTrace_.setPipeline(model_->getPtxCode());
      rayTrace_.setDomain(domain_);
      rayTrace_.setNumberOfRaysPerPoint(raysPerPoint_);
      rayTrace_.setUseRandomSeed(useRandomSeeds_);
      rayTrace_.setPeriodicBoundary(periodicBoundary_);
      for (auto &particle : model_->getParticleTypes()) {
        rayTrace_.insertNextParticle(particle);
      }
      numRates = rayTrace_.prepareParticlePrograms();
      fluxesIndexMap = IndexMap(rayTrace_.getParticles());
    }

    rayTrace_.updateSurface(); //  creates mesh
    auto numElements = rayTrace_.getNumberOfElements();

    // Initialize coverages
    if (!coveragesInitialized_)
      surfaceModel->initializeCoverages(diskMesh->nodes.size());
    auto coverages = surfaceModel->getCoverages(); // might be null
    const bool useCoverages = coverages != nullptr;

    if (useCoverages) {
      Logger::getInstance().addInfo("Using coverages.").print();

      Timer timer;
      Logger::getInstance().addInfo("Initializing coverages ... ").print();

      timer.start();
      for (size_t iterations = 1; iterations <= coverateIterations_;
           iterations++) {

        // get coverages and material ids in ray tracer
        const auto &materialIds =
            *diskMesh->getCellData().getScalarData("MaterialIds");

        CudaBuffer d_coverages; // device buffer for coverages and material ids
        translatePointToElementData(
            materialIds, coverages, d_coverages, transField,
            mesh); // transform point data to triangle mesh element data
        rayTrace_.setElementData(d_coverages,
                                 numCov + 1); // numCov + material ids

        // run the ray tracer
        rayTrace_.apply();

        // extract fluxes on points
        auto fluxes =
            SmartPointer<viennals::PointData<NumericType>>::New(); // flux on
                                                                   // point data
        translateElementToPointData(
            rayTrace_.getResults(), fluxes, fluxesIndexMap, elementKdTree,
            diskMesh, mesh); // transform element fluxes to point data

        // calculate coverages
        surfaceModel->updateCoverages(fluxes, materialIds);

        // output
        if (Logger::getLogLevel() >= 3) {
          assert(numElements == mesh->triangles.size());

          downloadCoverages(d_coverages, mesh->getCellData(), coverages,
                            numElements);

          rayTrace_.downloadResultsToPointData(mesh->getCellData());
          viennals::VTKWriter<NumericType>(
              mesh,
              name + "_covIinit_mesh_" + std::to_string(iterations) + ".vtp")
              .apply();

          insertReplaceScalarData(diskMesh->getCellData(), fluxes);
          insertReplaceScalarData(diskMesh->getCellData(), coverages);
          viennals::VTKWriter<NumericType>(
              diskMesh,
              name + "_covIinit_" + std::to_string(iterations) + ".vtp")
              .apply();
        }

        d_coverages.free();
        Logger::getInstance()
            .addInfo("Iteration: " + std::to_string(iterations))
            .print();
      }

      timer.finish();
      Logger::getInstance().addTiming("Coverage initialization", timer).print();
    }

    double previousTimeStep = 0.;
    size_t counter = 0;
    Timer rtTimer;
    Timer advTimer;
    while (remainingTime > 0.) {

      const auto &materialIds =
          *diskMesh->getCellData().getScalarData("MaterialIds");

      auto fluxes = SmartPointer<viennals::PointData<NumericType>>::New();
      CudaBuffer d_coverages; // device buffer for material ids and coverages
      rayTrace_.updateSurface();
      Logger::getInstance()
          .addDebug("Translating point to element data. Number of data: " +
                    std::to_string(numCov + 1))
          .print();
      translatePointToElementData(materialIds, coverages, d_coverages,
                                  transField, mesh);
      rayTrace_.setElementData(d_coverages, numCov + 1); // +1 material ids

      Logger::getInstance().addDebug("Running ray tracer ...").print();
      rtTimer.start();
      rayTrace_.apply();
      rtTimer.finish();

      Logger::getInstance()
          .addTiming("Top-down flux calculation", rtTimer)
          .print();

      // extract fluxes on points
      Logger::getInstance()
          .addDebug("Translating element to point data. Number of data: " +
                    std::to_string(numRates))
          .print();
      translateElementToPointData(rayTrace_.getResults(), fluxes,
                                  fluxesIndexMap, elementKdTree, diskMesh,
                                  mesh);

      auto velocities = surfaceModel->calculateVelocities(
          fluxes, diskMesh->nodes, materialIds);
      model_->getVelocityField()->setVelocities(velocities);
      assert(velocities->size() == pointKdTree->getNumberOfPoints());

      if (Logger::getLogLevel() >= 4) {
        if (useCoverages) {
          insertReplaceScalarData(diskMesh->getCellData(), coverages);
          downloadCoverages(d_coverages, mesh->getCellData(), coverages,
                            rayTrace_.getNumberOfElements());
        }

        insertReplaceScalarData(diskMesh->getCellData(), fluxes);
        rayTrace_.downloadResultsToPointData(mesh->getCellData());

        viennals::VTKWriter<NumericType>(
            mesh, name + "_mesh_" + std::to_string(counter) + ".vtp")
            .apply();

        if (velocities)
          insertReplaceScalarData(diskMesh->getCellData(), velocities,
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

      advTimer.start();
      advectionKernel.apply();
      advTimer.finish();

      Logger::getInstance().addTiming("Surface advection", advTimer).print();

      // update surface and move coverages to new surface
      diskMeshConv.apply();

      if (useCoverages)
        moveCoverages(transField, diskMesh, coverages);

      transField->buildKdTree(diskMesh->nodes);

      previousTimeStep = advectionKernel.getAdvectedTime();
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
  }

private:
  static void
  moveCoverages(SmartPointer<TranslationField<NumericType>> transField,
                SmartPointer<viennals::Mesh<NumericType>> points,
                SmartPointer<viennals::PointData<NumericType>> oldCoverages) {

    viennals::PointData<NumericType> newCoverages;

    // prepare new coverages
    for (unsigned i = 0; i < oldCoverages->getScalarDataSize(); i++) {
      std::vector<NumericType> tmp(points->nodes.size());
      newCoverages.insertNextScalarData(std::move(tmp),
                                        oldCoverages->getScalarDataLabel(i));
    }

    // move coverages
#pragma omp parallel for
    for (std::size_t i = 0; i < points->nodes.size(); i++) {
      auto &point = points->nodes[i];
      auto nearest = transField->getClosestPoint(point);

      for (unsigned j = 0; j < oldCoverages->getScalarDataSize(); j++) {
        assert(nearest->first < oldCoverages->getScalarData(j)->size());

        newCoverages.getScalarData(j)->at(i) =
            oldCoverages->getScalarData(j)->at(nearest->first);
      }
    }

    for (unsigned i = 0; i < oldCoverages->getScalarDataSize(); i++) {
      *oldCoverages->getScalarData(i) =
          std::move(*newCoverages.getScalarData(i));
    }
  }

  //   void smoothVelocities(
  //       SmartPointer<std::vector<NumericType>> velocities,
  //       SmartPointer<psKDTree<NumericType, std::array<float, 3>>> kdTree,
  //       const NumericType gridDelta) {
  //     const auto numPoints = velocities->size();
  //     std::cout << numPoints << " " << kdTree->getNumberOfPoints() <<
  //     std::endl; assert(numPoints == kdTree->getNumberOfPoints());

  //     std::vector<NumericType> smoothed(numPoints, 0.);

  // #pragma omp parallel for
  //     for (std::size_t i = 0; i < numPoints; ++i) {
  //       auto point = kdTree->getPoint(i);

  //       auto closePoints = kdTree->findNearestWithinRadius(point, gridDelta);

  //       unsigned n = 0;
  //       for (auto p : closePoints.value()) {
  //         smoothed[i] += velocities->at(p.first);
  //         n++;
  //       }

  //       if (n > 1) {
  //         smoothed[i] /= static_cast<NumericType>(n);
  //       }
  //     }

  //     *velocities = std::move(smoothed);
  //   }

  void
  downloadCoverages(CudaBuffer &d_coverages,
                    viennals::PointData<NumericType> &elementData,
                    SmartPointer<viennals::PointData<NumericType>> &coverages,
                    unsigned int numElements) {

    auto numCov = coverages->getScalarDataSize() + 1; // + material ids
    NumericType *temp = new NumericType[numElements * numCov];
    d_coverages.download(temp, numElements * numCov);

    // material IDs at front
    {
      auto matIds = elementData.getScalarData("Material");
      if (matIds == nullptr) {
        std::vector<NumericType> tmp(numElements);
        elementData.insertNextScalarData(std::move(tmp), "Material");
        matIds = elementData.getScalarData("Material");
      }
      if (matIds->size() != numElements)
        matIds->resize(numElements);
      std::memcpy(matIds->data(), temp, numElements * sizeof(NumericType));
    }

    for (unsigned i = 0; i < numCov - 1; i++) {
      auto covName = coverages->getScalarDataLabel(i);
      auto cov = elementData.getScalarData(covName);
      if (cov == nullptr) {
        std::vector<NumericType> covInit(numElements);
        elementData.insertNextScalarData(std::move(covInit), covName);
        cov = elementData.getScalarData(covName);
      }
      if (cov->size() != numElements)
        cov->resize(numElements);
      std::memcpy(cov->data(), temp + (i + 1) * numElements,
                  numElements * sizeof(NumericType));
    }

    delete temp;
  }

  void translateElementToPointData(
      CudaBuffer &d_elementData,
      SmartPointer<viennals::PointData<NumericType>> pointData,
      const IndexMap &indexMap,
      SmartPointer<KDTree<float, Vec3Df>> elementKdTree,
      SmartPointer<viennals::Mesh<NumericType>> pointMesh,
      SmartPointer<viennals::Mesh<float>> surfMesh) {

    const auto gridDelta = domain_->getGrid().getGridDelta();
    auto numData = indexMap.getNumberOfData();
    const auto &points = pointMesh->nodes;
    auto numPoints = points.size();
    auto numElements = elementKdTree->getNumberOfPoints();
    auto normals = pointMesh->cellData.getVectorData("Normals");
    auto elementNormals = surfMesh->cellData.getVectorData("Normals");

    // retrieve data from device
    std::vector<NumericType> elementData(numData * numElements);
    d_elementData.download(elementData.data(), numData * numElements);

    // prepare point data
    pointData->clear();
    for (const auto &label : indexMap) {
      std::vector<NumericType> data(numPoints, 0.);
      pointData->insertNextScalarData(std::move(data), label);
    }
    assert(pointData->getScalarDataSize() == numData); // assert number of data

#pragma omp parallel for
    for (unsigned i = 0; i < numPoints; i++) {

      auto closePoints = elementKdTree->findNearestWithinRadius(
          points[i], smoothFlux_ * smoothFlux_ * gridDelta *
                         gridDelta); // we have to use the squared distance here

      std::vector<NumericType> weights;
      weights.reserve(closePoints.value().size());
      NumericType sum = 0.;
      for (auto p : closePoints.value()) {
        assert(p.first < elementNormals->size());
        weights.push_back(
            DotProduct(normals->at(i), elementNormals->at(p.first)));
        if (weights.back() > 1e-6 && !std::isnan(weights.back()))
          sum += weights.back();
      }

      std::size_t nearestIdx = 0;
      if (sum <= 1e-6) {
        auto nearestPoint = elementKdTree->findNearest(points[i]);
        nearestIdx = nearestPoint->first;
      }

      for (unsigned j = 0; j < numData; j++) {
        NumericType value = 0.;
        if (sum > 1e-6) {
          unsigned n = 0;
          for (auto p : closePoints.value()) {
            if (weights[n] > 1e-6 && !std::isnan(weights[n])) {
              value += weights[n] * elementData[p.first + j * numElements];
            }
            n++;
          }
          value /= sum;
        } else {
          value = elementData[nearestIdx + j * numElements];
        }
        pointData->getScalarData(j)->at(i) = value;
      }
    }
  }

  template <class T>
  static void
  insertReplaceScalarData(viennals::PointData<T> &insertHere,
                          SmartPointer<viennals::PointData<T>> addData) {
    for (unsigned i = 0; i < addData->getScalarDataSize(); i++) {
      auto dataName = addData->getScalarDataLabel(i);
      auto data = insertHere.getScalarData(dataName, true);
      auto numElements = addData->getScalarData(i)->size();
      if (data == nullptr) {
        std::vector<NumericType> tmp(numElements);
        insertHere.insertNextScalarData(std::move(tmp), dataName);
        data = insertHere.getScalarData(dataName);
      }
      *data = *addData->getScalarData(i);
    }
  }

  template <class T>
  static void insertReplaceScalarData(viennals::PointData<T> &insertHere,
                                      SmartPointer<std::vector<T>> addData,
                                      std::string dataName) {
    auto data = insertHere.getScalarData(dataName, true);
    auto numElements = addData->size();
    if (data == nullptr) {
      std::vector<NumericType> tmp(numElements);
      insertHere.insertNextScalarData(std::move(tmp), dataName);
      data = insertHere.getScalarData(dataName);
    }
    *data = *addData;
  }

  static void translatePointToElementData(
      const std::vector<NumericType> &materialIds,
      SmartPointer<viennals::PointData<NumericType>> pointData,
      CudaBuffer &d_elementData,
      SmartPointer<TranslationField<NumericType>> translationField,
      SmartPointer<viennals::Mesh<float>> elementMesh) {

    auto numData = pointData ? pointData->getScalarDataSize() : 0;
    const auto &elements = elementMesh->template getElements<D>();
    auto numElements = elements.size();
    std::vector<NumericType> elementData((numData + 1) * numElements);
    assert(materialIds.size() == pointData->getScalarData(0)->size());

    auto closestPoints =
        SmartPointer<std::vector<NumericType>>::New(numElements);

#pragma omp parallel for
    for (unsigned i = 0; i < numElements; i++) {
      auto &elIdx = elements[i];
      std::array<NumericType, 3> elementCenter{
          (elementMesh->nodes[elIdx[0]][0] + elementMesh->nodes[elIdx[1]][0] +
           elementMesh->nodes[elIdx[2]][0]) /
              3.f,
          (elementMesh->nodes[elIdx[0]][1] + elementMesh->nodes[elIdx[1]][1] +
           elementMesh->nodes[elIdx[2]][1]) /
              3.f,
          (elementMesh->nodes[elIdx[0]][2] + elementMesh->nodes[elIdx[1]][2] +
           elementMesh->nodes[elIdx[2]][2]) /
              3.f};

      auto closestPoint = translationField->getClosestPoint(elementCenter);
      assert(closestPoint->first < materialIds.size());
      closestPoints->at(i) = closestPoint->first;

      // fill in material ids at front
      elementData[i] = materialIds[closestPoint->first];

      // scalar data with offset
      for (unsigned j = 0; j < numData; j++) {
        elementData[i + (j + 1) * numElements] =
            pointData->getScalarData(j)->at(closestPoint->first);
      }
    }

    insertReplaceScalarData(elementMesh->cellData, closestPoints, "pointIds");
    // for (int i = 0; i < numData; i++) {
    //   auto tmp = SmartPointer<std::vector<NumericType>>::New(
    //       elementData.begin() + (i + 1) * numElements,
    //       elementData.begin() + (i + 2) * numElements);
    //   insertReplaceScalarData(elementMesh->cellData, tmp,
    //                           pointData->getScalarDataLabel(i));
    // }
    // static int insert = 0;
    // viennals::VTKWriter<NumericType>(elementMesh, "insertElement_" +
    //                                           std::to_string(insert++) +
    //                                           ".vtp")
    //     .apply();

    d_elementData.alloc_and_upload(elementData);
  }

  bool checkInput() {
    if (!model_) {
      Logger::getInstance()
          .addWarning("No process model passed to Process.")
          .print();
      return true;
    }
    const auto name = model_->getProcessName();

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

    if (!model_->getPtxCode()) {
      Logger::getInstance()
          .addWarning("No pipeline in process model: " + name)
          .print();
      return true;
    }

    return false;
  }

  Context_t *context_;
  Tracer<NumericType, D> rayTrace_ = Tracer<NumericType, D>(context_);

  DomainType domain_;
  SmartPointer<ProcessModel<NumericType>> model_;

  NumericType processDuration_;
  long raysPerPoint_ = 1000;
  viennals::IntegrationSchemeEnum integrationScheme_ =
      viennals::IntegrationSchemeEnum::ENGQUIST_OSHER_1ST_ORDER;
  size_t coverateIterations_ = 20;

  NumericType smoothFlux_ = 1.;
  NumericType printTime_ = 0.;
  NumericType processTime_ = 0;

  bool useRandomSeeds_ = true;
  bool periodicBoundary_ = false;
  bool coveragesInitialized_ = false;
  bool rayTracerInitialized_ = false;
};

} // namespace gpu
} // namespace viennaps