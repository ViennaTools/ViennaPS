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
  using psDomainType = SmartPointer<::viennaps::Domain<NumericType, D>>;

public:
  Process(Context passedContext) : context(passedContext) {}

  void
  setProcessModel(SmartPointer<ProcessModel<NumericType>> passedProcessModel) {
    model = passedProcessModel;
  }

  void setDomain(psDomainType passedDomain) { domain = passedDomain; }

  void setProcessDuration(double duration) { processDuration = duration; }

  void setNumberOfRaysPerPoint(long numRays) { raysPerPoint = numRays; }

  void setMaxCoverageInitIterations(long maxIt) { maxIterations = maxIt; }

  void setPeriodicBoundary(const int passedPeriodic) {
    periodicBoundary = static_cast<bool>(passedPeriodic);
  }

  void setSmoothFlux(NumericType pSmoothFlux) { smoothFlux = pSmoothFlux; }

  void apply() {
    /* ---------- Process Setup --------- */
    if (!model) {
      Logger::getInstance()
          .addWarning("No process model passed to psProcess.")
          .print();
      return;
    }
    const auto name = model->getProcessName();

    if (!domain) {
      Logger::getInstance()
          .addWarning("No domain passed to psProcess.")
          .print();
      return;
    }

    if (!model->getSurfaceModel()) {
      Logger::getInstance()
          .addWarning("No surface model passed to Process.")
          .print();
      return;
    }

    Timer processTimer;
    processTimer.start();

    double remainingTime = processDuration;
    assert(domain->getLevelSets().size() != 0 && "No level sets in domain.");
    const NumericType gridDelta =
        domain->getLevelSets().back()->getGrid().getGridDelta();

    auto diskMesh = SmartPointer<viennals::Mesh<NumericType>>::New();
    viennals::ToDiskMesh<NumericType, D> diskMeshConv(diskMesh);

    /* --------- Setup advection kernel ----------- */
    viennals::Advect<NumericType, D> advectionKernel;
    advectionKernel.setIntegrationScheme(integrationScheme);
    // advectionKernel.setIgnoreVoids(true);

    auto transField = SmartPointer<TranslationField<NumericType>>::New(
        model->getVelocityField(), domain->getMaterialMap());
    advectionKernel.setVelocityField(transField);

    for (auto dom : domain->getLevelSets()) {
      advectionKernel.insertNextLevelSet(dom);
      diskMeshConv.insertNextLevelSet(dom);
    }

    /* --------- Setup element-point translation ----------- */

    auto mesh = rayTrace.getSurfaceMesh();
    auto elementKdTree =
        SmartPointer<KDTree<NumericType, std::array<NumericType, 3>>>::New();
    rayTrace.setKdTree(elementKdTree);

    diskMeshConv.apply();
    assert(diskMesh->nodes.size());

    transField->buildKdTree(diskMesh->nodes);

    /* --------- Setup for ray tracing ----------- */
    const bool useRayTracing = model->getParticleTypes().empty() ? false : true;
    unsigned int numRates = 0;
    unsigned int numCov = 0;
    IndexMap fluxesIndexMap;

    if (useRayTracing && !rayTracerInitialized) {
      if (!model->getPtxCode()) {
        Logger::getInstance()
            .addWarning("No pipeline in process model. Aborting.")
            .print();
        return;
      }
      rayTrace.setPipeline(model->getPtxCode());
      rayTrace.setLevelSet(domain);
      rayTrace.setNumberOfRaysPerPoint(raysPerPoint);
      rayTrace.setUseRandomSeed(useRandomSeeds);
      rayTrace.setPeriodicBoundary(periodicBoundary);
      for (auto &particle : model->getParticleTypes()) {
        rayTrace.insertNextParticle(particle);
      }
      numRates = rayTrace.prepareParticlePrograms();
      fluxesIndexMap = IndexMap(rayTrace.getParticles());
    }

    // Determine whether there are process parameters used in ray tracing
    model->getSurfaceModel()->initializeProcessParameters();
    const bool useProcessParams =
        model->getSurfaceModel()->getProcessParameters() != nullptr;

    if (useProcessParams)
      Logger::getInstance().addInfo("Using process parameters.").print();

    unsigned int numElements = 0;
    if (useRayTracing) {
      rayTrace.updateSurface(); // also creates mesh
      numElements = rayTrace.getNumberOfElements();
    }

    // Initialize coverages
    if (!coveragesInitialized)
      model->getSurfaceModel()->initializeCoverages(diskMesh->nodes.size());
    auto coverages = model->getSurfaceModel()->getCoverages(); // might be null
    const bool useCoverages = coverages != nullptr;

    if (useCoverages) {
      numCov = coverages->getScalarDataSize();
      rayTrace.setUseCellData(numCov + 1); // + material IDs

      Logger::getInstance().addInfo("Using coverages.").print();

      Timer timer;
      Logger::getInstance().addInfo("Initializing coverages ... ").print();

      timer.start();
      for (size_t iterations = 1; iterations <= maxIterations; iterations++) {

        // get coverages and material ids in ray tracer
        const auto &materialIds =
            *diskMesh->getCellData().getScalarData("MaterialIds");
        CudaBuffer d_coverages; // device buffer
        translatePointToElementData(materialIds, coverages, d_coverages,
                                    transField, mesh);
        rayTrace.setCellData(d_coverages, numCov + 1); // + material ids

        // run the ray tracer
        rayTrace.apply();

        // extract fluxes on points
        auto fluxes = SmartPointer<viennals::PointData<NumericType>>::New();
        translateElementToPointData(rayTrace.getResults(), fluxes,
                                    fluxesIndexMap, elementKdTree, diskMesh,
                                    mesh, gridDelta);

        // calculate coverages
        model->getSurfaceModel()->updateCoverages(fluxes, materialIds);

        if (Logger::getLogLevel() >= 3) {
          assert(numElements == mesh->triangles.size());

          downloadCoverages(d_coverages, mesh->getCellData(), coverages,
                            numElements);

          rayTrace.downloadResultsToPointData(mesh->getCellData());
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
    Timer callbackTimer;
    Timer advTimer;
    while (remainingTime > 0.) {
      Logger::getInstance()
          .addInfo("Remaining time: " + std::to_string(remainingTime))
          .print();

      const auto &materialIds =
          *diskMesh->getCellData().getScalarData("MaterialIds");

      auto fluxes = SmartPointer<viennals::PointData<NumericType>>::New();
      CudaBuffer d_coverages; // device buffer for material ids and coverages
      if (useRayTracing) {
        rayTrace.updateSurface();
        translatePointToElementData(materialIds, coverages, d_coverages,
                                    transField, mesh);
        rayTrace.setCellData(d_coverages, numCov + 1); // +1 material ids

        rtTimer.start();
        rayTrace.apply();
        rtTimer.finish();

        // extract fluxes on points
        translateElementToPointData(rayTrace.getResults(), fluxes,
                                    fluxesIndexMap, elementKdTree, diskMesh,
                                    mesh, gridDelta);

        Logger::getInstance()
            .addTiming("Top-down flux calculation", rtTimer)
            .print();
      }

      auto velocities = model->getSurfaceModel()->calculateVelocities(
          fluxes, diskMesh->nodes, materialIds);
      model->getVelocityField()->setVelocities(velocities);
      assert(velocities->size() == pointKdTree->getNumberOfPoints());

      if (Logger::getLogLevel() >= 4) {
        if (useCoverages) {
          insertReplaceScalarData(diskMesh->getCellData(), coverages);
          downloadCoverages(d_coverages, mesh->getCellData(), coverages,
                            rayTrace.getNumberOfElements());
        }

        if (useRayTracing) {
          insertReplaceScalarData(diskMesh->getCellData(), fluxes);
          rayTrace.downloadResultsToPointData(mesh->getCellData());

          viennals::VTKWriter<NumericType>(
              mesh, name + "_mesh_" + std::to_string(counter) + ".vtp")
              .apply();
        }

        if (velocities)
          insertReplaceScalarData(diskMesh->getCellData(), velocities,
                                  "velocities");

        viennals::VTKWriter<NumericType>(
            diskMesh, name + "_" + std::to_string(counter) + ".vtp")
            .apply();

        counter++;
      }

      if (useRayTracing)
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
    }

    processTime = processDuration - remainingTime;
    processTimer.finish();

    Logger::getInstance()
        .addTiming("\nProcess " + name, processTimer)
        .addTiming("Surface advection total time",
                   advTimer.totalDuration * 1e-9,
                   processTimer.totalDuration * 1e-9)
        .print();
    if (useRayTracing) {
      Logger::getInstance()
          .addTiming("Top-down flux calculation total time",
                     rtTimer.totalDuration * 1e-9,
                     processTimer.totalDuration * 1e-9)
          .print();
    }
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
      SmartPointer<KDTree<float, std::array<float, 3>>> elementKdTree,
      SmartPointer<viennals::Mesh<NumericType>> pointMesh,
      SmartPointer<viennals::Mesh<float>> surfMesh,
      const NumericType gridDelta) {

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
          points[i], smoothFlux * gridDelta);

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
      auto data = insertHere.getScalarData(dataName);
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
    auto data = insertHere.getScalarData(dataName);
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

  Context_t *context;
  curtTracer<NumericType, D> rayTrace = curtTracer<NumericType, D>(context);

  psDomainType domain = nullptr;
  SmartPointer<ProcessModel<NumericType>> model = nullptr;
  viennals::IntegrationSchemeEnum integrationScheme =
      viennals::IntegrationSchemeEnum::ENGQUIST_OSHER_1ST_ORDER;
  NumericType processDuration;
  long raysPerPoint = 1000;
  bool useRandomSeeds = true;
  size_t maxIterations = 20;
  bool periodicBoundary = false;
  bool coveragesInitialized = false;
  bool rayTracerInitialized = false;
  NumericType smoothFlux = 1.;
  NumericType printTime = 0.;
  NumericType processTime = 0;
};

} // namespace gpu
} // namespace viennaps