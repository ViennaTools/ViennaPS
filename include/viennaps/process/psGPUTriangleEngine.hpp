#pragma once

#ifdef VIENNACORE_COMPILE_GPU

#include "../psCreateSurfaceMesh.hpp"
#include "../psDomain.hpp"
#include "../psElementToPointData.hpp"
#include "../psPointToElementData.hpp"
#include "psDesorptionSource.hpp"
#include "psFluxEngine.hpp"
#include "psProcessModel.hpp"

#include <vcContext.hpp>

#include <lsMesh.hpp>

#include <gpu/raygTraceTriangle.hpp>

#include <cassert>
#include <cmath>

namespace viennaps {

using namespace viennacore;

VIENNAPS_TEMPLATE_ND(NumericType, D)
class GPUTriangleEngine final : public FluxEngine<NumericType, D> {
  using KDTreeType =
      SmartPointer<KDTree<NumericType, std::array<NumericType, 3>>>;
  using MeshType = SmartPointer<viennals::Mesh<float>>;
  using PostProcessingType =
      ElementToPointData<NumericType, float, viennaray::gpu::ResultType, true,
                         D == 3>;

public:
  explicit GPUTriangleEngine(std::shared_ptr<DeviceContext> deviceContext)
      : deviceContext_(deviceContext), rayTracer_(deviceContext) {
    surfaceMesh_ = MeshType::New();
    elementKdTree_ = KDTreeType::New();
  }

  ProcessResult checkInput(ProcessContext<NumericType, D> &context) override {

    auto model =
        std::dynamic_pointer_cast<gpu::ProcessModelGPU<NumericType, D>>(
            context.model);
    if (!model) {
      VIENNACORE_LOG_WARNING("Invalid GPU process model.");
      return ProcessResult::INVALID_INPUT;
    }

    const auto name = context.model->getProcessName().value_or("default");
    if (model->getParticleTypes().empty()) {
      VIENNACORE_LOG_WARNING("No particle types in process model: " + name);
      return ProcessResult::INVALID_INPUT;
    }

    if (model->getCallableFileName().empty()) {
      VIENNACORE_LOG_WARNING("No callables in process model: " + name);
      return ProcessResult::INVALID_INPUT;
    }

    model_ = model;

    return ProcessResult::SUCCESS;
  }

  ProcessResult initialize(ProcessContext<NumericType, D> &context) override {
    assert(model_ && "Model not set.");
    if (!rayTracerInitialized_) {
      rayTracer_.setParticleCallableMap(model_->getParticleCallableMap());
      rayTracer_.setCallables(model_->getCallableFileName(),
                              deviceContext_->modulePath);
      rayTracer_.setNumberOfRaysPerPoint(context.rayTracingParams.raysPerPoint);
      rayTracer_.setUseRandomSeeds(context.rayTracingParams.useRandomSeeds);
      rayTracer_.setMaxBoundaryHits(context.rayTracingParams.maxBoundaryHits);
      if (context.rayTracingParams.maxReflections > 0)
        rayTracer_.setMaxReflections(context.rayTracingParams.maxReflections);
      if (!context.rayTracingParams.useRandomSeeds)
        rayTracer_.setRngSeed(context.rayTracingParams.rngSeed);
      for (auto &particle : model_->getParticleTypes()) {
        rayTracer_.insertNextParticle(particle);
      }

      // Check boundary conditions
      if (context.rayTracingParams.ignoreFluxBoundaries) {
        rayTracer_.setIgnoreBoundary(true);
      } else if (context.flags.domainHasPeriodicBoundaries) {
        rayTracer_.setPeriodicBoundary(true);
      }

      rayTracer_.prepareParticlePrograms();
    }
    rayTracer_.setParameters(model_->getProcessDataDPtr());
    rayTracerInitialized_ = true;

    return ProcessResult::SUCCESS;
  }

  ProcessResult
  updateSurface(ProcessContext<NumericType, D> &context) override {
    this->timer_.start();

    assert(surfaceMesh_ && "Surface mesh not initialized.");
    assert(elementKdTree_ && "Element KDTree not initialized.");
    CreateSurfaceMesh<NumericType, float, D>(
        context.domain->getSurface(), surfaceMesh_, elementKdTree_, 1e-12,
        context.rayTracingParams.minNodeDistanceFactor)
        .apply();
    assert(!surfaceMesh_->nodes.empty());

    viennaray::TriangleMesh triangleMesh;

    if constexpr (D == 2) {
      viennaray::LineMesh lineMesh;
      lineMesh.gridDelta = static_cast<float>(context.domain->getGridDelta());
      lineMesh.lines = std::move(surfaceMesh_->lines);
      lineMesh.nodes = std::move(surfaceMesh_->nodes);
      lineMesh.minimumExtent = surfaceMesh_->minimumExtent;
      lineMesh.maximumExtent = surfaceMesh_->maximumExtent;
      assert(!lineMesh.lines.empty());

      triangleMesh = convertLinesToTriangles(lineMesh);
      assert(!triangleMesh.triangles.empty());

      std::vector<Vec3D<NumericType>> triangleCenters;
      triangleCenters.reserve(triangleMesh.triangles.size());
      for (const auto &tri : triangleMesh.triangles) {
        Vec3D<NumericType> center = {0, 0, 0};
        for (int i = 0; i < 3; ++i) {
          center[0] += triangleMesh.nodes[tri[i]][0];
          center[1] += triangleMesh.nodes[tri[i]][1];
          center[2] += triangleMesh.nodes[tri[i]][2];
        }
        triangleCenters.push_back(center / static_cast<NumericType>(3.0));
      }
      assert(triangleCenters.size() > 0);
      elementKdTree_->setPoints(triangleCenters);
      elementKdTree_->build();
    } else {
      CopyTriangleMesh(static_cast<float>(context.domain->getGridDelta()),
                       surfaceMesh_, triangleMesh);
      // preserves surfaceMesh_
    }

    rayTracer_.setGeometry(triangleMesh);

    if constexpr (D == 2) {
      surfaceMesh_->clear();
      surfaceMesh_->nodes = std::move(triangleMesh.nodes);
      surfaceMesh_->triangles = std::move(triangleMesh.triangles);
      surfaceMesh_->getCellData().insertReplaceVectorData(
          std::move(triangleMesh.normals), "Normals");
      surfaceMesh_->minimumExtent = triangleMesh.minimumExtent;
      surfaceMesh_->maximumExtent = triangleMesh.maximumExtent;
    }

    if (model_->useMaterialIds()) {
      auto const &pointMaterialIds =
          *context.diskMesh->getCellData().getScalarData("MaterialIds");
      std::vector<int> elementMaterialIds;
      auto &pointKdTree = context.translationField->getKdTree();
      if (!pointKdTree) {
        pointKdTree = KDTreeType::New();
        context.translationField->setKdTree(pointKdTree);
      }
      if (pointKdTree->getNumberOfPoints() != context.diskMesh->nodes.size()) {
        pointKdTree->setPoints(context.diskMesh->nodes);
        pointKdTree->build();
      }
      PointToElementDataSingle<NumericType, NumericType, int, float>(
          pointMaterialIds, elementMaterialIds, *pointKdTree, surfaceMesh_)
          .apply();
      rayTracer_.setMaterialIds(elementMaterialIds);
    }

    assert(context.diskMesh->nodes.size() > 0);
    assert(!surfaceMesh_->nodes.empty());
    this->timer_.finish();

    return ProcessResult::SUCCESS;
  }

  ProcessResult calculateFluxes(
      ProcessContext<NumericType, D> &context,
      SmartPointer<viennals::PointData<NumericType>> &fluxes) override {

    this->timer_.start();

    CudaBuffer d_coverages; // device buffer for coverages
    if (context.flags.useCoverages) {
      auto coverages = model_->getSurfaceModel()->getCoverages();
      assert(coverages);
      assert(context.diskMesh);
      assert(context.translationField);
      auto numCov = coverages->getScalarDataSize();
      auto &pointKdTree = context.translationField->getKdTree();
      if (!pointKdTree) {
        pointKdTree = KDTreeType::New();
        context.translationField->setKdTree(pointKdTree);
      }
      if (pointKdTree->getNumberOfPoints() != context.diskMesh->nodes.size()) {
        pointKdTree->setPoints(context.diskMesh->nodes);
        pointKdTree->build();
      }
      gpu::PointToElementData<NumericType, float>(d_coverages, coverages,
                                                  *pointKdTree, surfaceMesh_)
          .apply();
      rayTracer_.setElementData(d_coverages, numCov);
    }

    std::vector<NumericType> desorptionWeights;
    if (auto materialIds =
            context.diskMesh->getCellData().getScalarData("MaterialIds")) {
      desorptionWeights =
          model_->getSurfaceModel()->getDesorptionWeights(*materialIds);
    }

    // run the ray tracer
    rayTracer_.clearSurfaceSource();
    rayTracer_.apply(); // device detach point here
    ++this->fluxCalculationsCount_;

    // Prepare post-processing
    PostProcessingType postProcessing(
        model_->getParticleDataLabels(), fluxes, elementKdTree_,
        context.diskMesh, surfaceMesh_,
        context.domain->getGridDelta() *
            (context.rayTracingParams.smoothingNeighbors + 1));
    postProcessing.prepare();

    rayTracer_.normalizeResults(); // device sync point here
    auto combinedResults = rayTracer_.getResults();
    if (desorptionWeights.size() == context.diskMesh->getNodes().size()) {
      addDesorptionFlux(context, desorptionWeights, combinedResults);
    } else if (desorptionWeights.size() > 0) {
      VIENNACORE_LOG_WARNING("Desorption weights size does not match number "
                             "of mesh nodes. Skipping desorption flux.");
    }

    // always persist per-triangle flux so callers can access it
    saveResultsToPointData(surfaceMesh_->getCellData(), combinedResults);
    context.triangleMesh = surfaceMesh_;

    postProcessing.setElementDataArrays(std::move(combinedResults));
    postProcessing.convert(); // run post-processing

    if (Logger::hasIntermediate()) {
      if (context.flags.useCoverages) {
        auto coverages = model_->getSurfaceModel()->getCoverages();
        downloadCoverages(d_coverages, surfaceMesh_->getCellData(), coverages,
                          surfaceMesh_->getElements<3>().size());
      }
      viennals::VTKWriter<float>(
          surfaceMesh_, context.intermediateOutputPath +
                            context.getProcessName() + "_flux_" +
                            std::to_string(context.currentIteration) + ".vtp")
          .apply();
    }

    d_coverages.free();
    this->timer_.finish();

    return ProcessResult::SUCCESS;
  }

private:
  static void
  downloadCoverages(CudaBuffer &d_coverages,
                    viennals::PointData<float> &elementData,
                    SmartPointer<viennals::PointData<NumericType>> &coverages,
                    unsigned int numElements) {

    auto numCov = coverages->getScalarDataSize();
    auto *temp = new float[numElements * numCov];
    d_coverages.download(temp, numElements * numCov);

    for (unsigned i = 0; i < numCov; i++) {
      auto covName = coverages->getScalarDataLabel(i);
      std::vector<float> values(numElements);
      std::memcpy(values.data(), &temp[i * numElements],
                  numElements * sizeof(float));
      elementData.insertReplaceScalarData(values, covName);
    }

    delete[] temp;
  }

  void saveResultsToPointData(
      viennals::PointData<float> &pointData,
      const std::vector<std::vector<viennaray::gpu::ResultType>> &results) {
    const auto numPoints = rayTracer_.getNumberOfElements();
    assert(numPoints == surfaceMesh_->getElements<3>().size());
    auto particles = rayTracer_.getParticles();
    const auto &dataLabels = model_->getParticleDataLabels();

    assert(dataLabels.size() == results.size());

    for (int dIdx = 0; dIdx < dataLabels.size(); dIdx++) {
      const auto &name = dataLabels[dIdx];
      const auto &data = results[dIdx];

      std::vector<float> values(numPoints);
      for (unsigned i = 0; i < numPoints; ++i) {
        values[i] = static_cast<float>(data[i]);
      }

      pointData.insertReplaceScalarData(std::move(values), name);
    }
  }

  void addDesorptionFlux(
      ProcessContext<NumericType, D> &context,
      const std::vector<NumericType> &diskDesorptionWeights,
      std::vector<std::vector<viennaray::gpu::ResultType>> &combinedResults) {
    auto &pointKdTree = context.translationField->getKdTree();
    if (!pointKdTree) {
      pointKdTree = KDTreeType::New();
      context.translationField->setKdTree(pointKdTree);
    }
    if (pointKdTree->getNumberOfPoints() != context.diskMesh->nodes.size()) {
      pointKdTree->setPoints(context.diskMesh->nodes);
      pointKdTree->build();
    }

    std::vector<float> elementWeights;
    PointToElementDataSingle<NumericType, NumericType, float, float>(
        diskDesorptionWeights, elementWeights, *pointKdTree, surfaceMesh_)
        .apply();

    const auto &triangles = surfaceMesh_->triangles;
    const auto &nodes = surfaceMesh_->nodes;
    const auto meshNormals =
        surfaceMesh_->getCellData().getVectorData("Normals");
    const std::vector<Vec3Df> emptyNormals;
    const auto &normals = meshNormals != nullptr ? *meshNormals : emptyNormals;

    auto sourceData = makeTriangleDesorptionSourceData<float>(
        nodes, triangles, normals, elementWeights,
        static_cast<float>(context.domain->getGridDelta()));
    if (!sourceData.hasSource) {
      // No active desorption sources, skip ray tracing
      return;
    }

    rayTracer_.setSurfaceSource(sourceData.positions, sourceData.normals,
                                sourceData.weights, sourceData.sourceArea,
                                sourceData.sourceOffset);
    rayTracer_.apply();
    rayTracer_.normalizeResults();
    auto desorptionResults = rayTracer_.getResults();
    rayTracer_.clearSurfaceSource();

    for (std::size_t dataIdx = 0; dataIdx < combinedResults.size(); ++dataIdx) {
      for (std::size_t i = 0; i < combinedResults[dataIdx].size(); ++i) {
        combinedResults[dataIdx][i] += desorptionResults[dataIdx][i];
      }
    }
  }

private:
  std::shared_ptr<DeviceContext> deviceContext_;
  viennaray::gpu::TraceTriangle<NumericType, D> rayTracer_;
  SmartPointer<gpu::ProcessModelGPU<NumericType, D>> model_;

  KDTreeType elementKdTree_;
  MeshType surfaceMesh_;

  bool rayTracerInitialized_ = false;
};

} // namespace viennaps

#endif
