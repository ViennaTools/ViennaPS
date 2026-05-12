#pragma once

#include "../psCreateSurfaceMesh.hpp"
#include "../psElementToPointData.hpp"
#include "../psPointToElementData.hpp"

#include "psDesorptionSource.hpp"
#include "psFluxEngine.hpp"
#include "psProcessModel.hpp"

#include <rayTraceTriangle.hpp>

#include <cmath>

namespace viennaps {

using namespace viennacore;

VIENNAPS_TEMPLATE_ND(NumericType, D)
class CPUTriangleEngine final : public FluxEngine<NumericType, D> {
  using KDTreeType =
      SmartPointer<KDTree<NumericType, std::array<NumericType, 3>>>;
  using MeshType = SmartPointer<viennals::Mesh<float>>;
  using PostProcessingType =
      ElementToPointData<NumericType, float, NumericType, true, D == 3>;

public:
  ProcessResult checkInput(ProcessContext<NumericType, D> &context) override {
    auto model = std::dynamic_pointer_cast<ProcessModelCPU<NumericType, D>>(
        context.model);
    if (!model) {
      VIENNACORE_LOG_WARNING("Invalid process model.");
      return ProcessResult::INVALID_INPUT;
    }
    model_ = model;
    return ProcessResult::SUCCESS;
  }

  ProcessResult initialize(ProcessContext<NumericType, D> &context) override {
    assert(model_ != nullptr);
    // Map the domain boundary to the ray tracing boundaries
    viennaray::BoundaryCondition rayBoundaryCondition[D];
    if (context.rayTracingParams.ignoreFluxBoundaries) {
      for (unsigned i = 0; i < D; ++i)
        rayBoundaryCondition[i] = viennaray::BoundaryCondition::IGNORE_BOUNDARY;
    } else {
      for (unsigned i = 0; i < D; ++i)
        rayBoundaryCondition[i] = util::convertBoundaryCondition(
            context.domain->getGrid().getBoundaryConditions(i));
    }
    if constexpr (D == 2) {
      rayTracer_.setSourceDirection(viennaray::TraceDirection::POS_Y);
    } else {
      rayTracer_.setSourceDirection(viennaray::TraceDirection::POS_Z);
    }
    rayTracer_.setBoundaryConditions(rayBoundaryCondition);
    rayTracer_.setNumberOfRaysPerPoint(context.rayTracingParams.raysPerPoint);
    rayTracer_.setMaxBoundaryHits(context.rayTracingParams.maxBoundaryHits);
    if (context.rayTracingParams.maxReflections > 0)
      rayTracer_.setMaxReflections(context.rayTracingParams.maxReflections);
    rayTracer_.setUseRandomSeeds(context.rayTracingParams.useRandomSeeds);
    if (!context.rayTracingParams.useRandomSeeds)
      rayTracer_.setRngSeed(context.rayTracingParams.rngSeed);

    if (auto source = model_->getSource()) {
      rayTracer_.setSource(source);
      VIENNACORE_LOG_INFO("Using custom source.");
    }
    if (auto primaryDirection = model_->getPrimaryDirection()) {
      VIENNACORE_LOG_INFO("Using primary direction: " +
                          util::arrayToString(primaryDirection.value()));
      rayTracer_.setPrimaryDirection(primaryDirection.value());
    }

    model_->initializeParticleDataLogs();

    return ProcessResult::SUCCESS;
  }

  ProcessResult
  updateSurface(ProcessContext<NumericType, D> &context) override {
    this->timer_.start();
    if (!surfaceMesh_)
      surfaceMesh_ = MeshType::New();
    if (!elementKdTree_)
      elementKdTree_ = KDTreeType::New();

    CreateSurfaceMesh<NumericType, float, D>(
        context.domain->getLevelSets().back(), surfaceMesh_, elementKdTree_,
        1e-12, context.rayTracingParams.minNodeDistanceFactor)
        .apply();

    viennaray::TriangleMesh triangleMesh;

    if constexpr (D == 2) {
      viennaray::LineMesh lineMesh;
      lineMesh.gridDelta = static_cast<float>(context.domain->getGridDelta());
      lineMesh.lines = std::move(surfaceMesh_->lines);
      lineMesh.nodes = std::move(surfaceMesh_->nodes);
      lineMesh.minimumExtent = surfaceMesh_->minimumExtent;
      lineMesh.maximumExtent = surfaceMesh_->maximumExtent;

      triangleMesh = convertLinesToTriangles(lineMesh);
      assert(triangleMesh.triangles.size() > 0);

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
    }

    rayTracer_.setGeometry(triangleMesh);

    if constexpr (D == 2) {
      surfaceMesh_->nodes = std::move(triangleMesh.nodes);
      surfaceMesh_->triangles = std::move(triangleMesh.triangles);
      surfaceMesh_->getCellData().insertReplaceVectorData(
          std::move(triangleMesh.normals), "Normals");
      surfaceMesh_->minimumExtent = triangleMesh.minimumExtent;
      surfaceMesh_->maximumExtent = triangleMesh.maximumExtent;
    }

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

    assert(context.diskMesh->nodes.size() > 0);
    assert(!surfaceMesh_->nodes.empty());
    this->timer_.finish();

    return ProcessResult::SUCCESS;
  }

  ProcessResult calculateSourceFluxes(
      ProcessContext<NumericType, D> &context,
      SmartPointer<viennals::PointData<NumericType>> &fluxes) override {
    this->timer_.start();
    viennaray::TracingData<NumericType> rayTracingData;
    auto surfaceModel = context.model->getSurfaceModel();
    std::vector<NumericType> desorptionWeights;

    // copy coverages to the ray tracer
    if (context.flags.useCoverages) {
      if (auto materialIds =
              context.diskMesh->getCellData().getScalarData("MaterialIds")) {
        desorptionWeights = surfaceModel->getDesorptionWeights(*materialIds)
                                .value_or(std::vector<NumericType>{});
      }

      auto &pointKdTree = context.translationField->getKdTree();
      if (!pointKdTree) {
        pointKdTree = KDTreeType::New();
        context.translationField->setKdTree(pointKdTree);
      }
      if (pointKdTree->getNumberOfPoints() != context.diskMesh->nodes.size()) {
        pointKdTree->setPoints(context.diskMesh->nodes);
        pointKdTree->build();
      }
      // Coverages are copied to elementData so there is no need to move them
      // back to the model
      viennals::PointData<NumericType> elementData;
      PointToElementData<NumericType, float>(
          elementData, surfaceModel->getCoverages(), *pointKdTree, surfaceMesh_,
          Logger::hasIntermediate())
          .apply();
      // Move data from PointData to TracingData
      rayTracingData = movePointDataToRayData(elementData);
    }

    if (context.flags.useProcessParams) {
      // store scalars in addition to coverages
      auto processParams = surfaceModel->getProcessParameters();
      NumericType numParams = processParams->getScalarData().size();
      rayTracingData.setNumberOfScalarData(numParams);
      for (size_t i = 0; i < numParams; ++i) {
        rayTracingData.setScalarData(i, processParams->getScalarData(i),
                                     processParams->getScalarDataLabel(i));
      }
    }

    if (context.flags.useCoverages || context.flags.useProcessParams)
      rayTracer_.setGlobalData(rayTracingData);

    runRayTracer(context, fluxes, desorptionWeights);

    // output
    if (Logger::hasIntermediate() && D == 3) {
      viennals::VTKWriter<float>(
          surfaceMesh_, context.intermediateOutputPath +
                            context.getProcessName() + "_flux_" +
                            std::to_string(context.currentIteration) + ".vtp")
          .apply();
    }

    this->timer_.finish();

    return ProcessResult::SUCCESS;
  }

  ProcessResult calculateSurfaceFluxes(
      ProcessContext<NumericType, D> &,
      SmartPointer<viennals::PointData<NumericType>> &) override {
    return ProcessResult::NOT_IMPLEMENTED;
  }

private:
  void runRayTracer(ProcessContext<NumericType, D> &context,
                    SmartPointer<viennals::PointData<NumericType>> &fluxes,
                    const std::vector<NumericType> &desorptionWeights) {
    assert(fluxes != nullptr);
    fluxes->clear();

    std::vector<std::vector<NumericType>> elementFluxes;
    std::vector<std::string> elementFluxLabels;

    unsigned particleIdx = 0;
    for (auto &particle : model_->getParticleTypes()) {
      int dataLogSize = model_->getParticleLogSize(particleIdx);
      if (dataLogSize > 0) {
        rayTracer_.getDataLog().data.resize(1);
        rayTracer_.getDataLog().data[0].resize(dataLogSize, 0.);
      }
      rayTracer_.setParticleType(particle);
      rayTracer_.apply();
      ++this->fluxCalculationsCount_;

      auto info = rayTracer_.getRayTraceInfo();

      VIENNACORE_LOG_DEBUG(
          "Particle " + std::to_string(particleIdx) +
          "\n\tRays Traced: " + std::to_string(info.totalRaysTraced) +
          "\n\tNon-Geometry Hits: " + std::to_string(info.nonGeometryHits) +
          "\n\tGeometry Hits: " + std::to_string(info.geometryHits) +
          "\n\tParticle Hits: " + std::to_string(info.particleHits) +
          (info.warning ? "\n\tWarning during ray tracing."
                        : (info.error ? "\n\tError during ray tracing." : "")));

      // fill up fluxes vector with fluxes from this particle type
      auto &localData = rayTracer_.getLocalData();
      int numFluxes = particle->getLocalDataLabels().size();
      std::vector<std::vector<NumericType>> particleFluxes;
      particleFluxes.reserve(numFluxes);
      std::vector<std::string> particleFluxLabels;
      particleFluxLabels.reserve(numFluxes);
      for (int i = 0; i < numFluxes; ++i) {
        auto flux = std::move(localData.getVectorData(i));

        // normalize
        rayTracer_.normalizeFlux(flux,
                                 context.rayTracingParams.normalizationType);

        particleFluxLabels.push_back(localData.getVectorDataLabel(i));
        particleFluxes.push_back(std::move(flux));
      }

      if (desorptionWeights.size() == context.diskMesh->getNodes().size()) {
        addDesorptionFlux(context, desorptionWeights, particleFluxes,
                          numFluxes);
      } else if (desorptionWeights.size() > 0) {
        VIENNACORE_LOG_WARNING("Desorption weights size does not match number "
                               "of mesh nodes. Skipping desorption flux.");
      }

      for (int i = 0; i < numFluxes; ++i) {
        elementFluxLabels.push_back(std::move(particleFluxLabels[i]));
        elementFluxes.push_back(std::move(particleFluxes[i]));
      }

      model_->mergeParticleData(rayTracer_.getDataLog(), particleIdx);
      ++particleIdx;
    }

    saveElementFluxesToTriangleMesh(elementFluxes, elementFluxLabels);
    context.triangleMesh = surfaceMesh_;

    // map fluxes to points
    PostProcessingType postProcessing(
        model_->getParticleDataLabels(), fluxes, elementKdTree_,
        context.diskMesh, surfaceMesh_,
        context.domain->getGridDelta() *
            (context.rayTracingParams.smoothingNeighbors + 1));
    postProcessing.setElementDataArrays(std::move(elementFluxes));
    postProcessing.apply();
  }

  void addDesorptionFlux(ProcessContext<NumericType, D> &context,
                         const std::vector<NumericType> &diskDesorptionWeights,
                         std::vector<std::vector<NumericType>> &particleFluxes,
                         int numFluxes) {
    auto &pointKdTree = context.translationField->getKdTree();
    if (!pointKdTree) {
      pointKdTree = KDTreeType::New();
      context.translationField->setKdTree(pointKdTree);
    }
    if (pointKdTree->getNumberOfPoints() != context.diskMesh->nodes.size()) {
      pointKdTree->setPoints(context.diskMesh->nodes);
      pointKdTree->build();
    }

    std::vector<NumericType> elementWeights;
    PointToElementDataSingle<NumericType, NumericType, NumericType, float>(
        diskDesorptionWeights, elementWeights, *pointKdTree, surfaceMesh_)
        .apply();

    const auto &triangles = surfaceMesh_->triangles;
    const auto &nodes = surfaceMesh_->nodes;
    const auto meshNormals =
        surfaceMesh_->getCellData().getVectorData("Normals");
    const std::vector<Vec3Df> emptyNormals;
    const auto &normals = meshNormals != nullptr ? *meshNormals : emptyNormals;

    auto sourceData = makeTriangleDesorptionSourceData<NumericType>(
        nodes, triangles, normals, elementWeights,
        static_cast<NumericType>(context.domain->getGridDelta()));
    if (!sourceData.hasSource) {
      return;
    }

    auto source = std::make_shared<DesorptionSource<NumericType, D>>(
        std::move(sourceData), context.rayTracingParams.raysPerPoint);

    rayTracer_.setSource(source);
    rayTracer_.apply();
    ++this->fluxCalculationsCount_;

    auto &desorptionData = rayTracer_.getLocalData();
    for (int i = 0; i < numFluxes; ++i) {
      auto desorptionFlux = std::move(desorptionData.getVectorData(i));
      rayTracer_.normalizeFlux(desorptionFlux,
                               context.rayTracingParams.normalizationType);

      if (desorptionFlux.size() == particleFluxes[i].size()) {
#pragma omp parallel for
        for (std::size_t j = 0; j < desorptionFlux.size(); ++j) {
          particleFluxes[i][j] += desorptionFlux[j];
        }
      }
    }

    if (auto modelSource = model_->getSource()) {
      rayTracer_.setSource(modelSource);
    } else {
      rayTracer_.resetSource();
    }
  }

  void saveElementFluxesToTriangleMesh(
      const std::vector<std::vector<NumericType>> &elementFluxes,
      const std::vector<std::string> &elementFluxLabels) {
    assert(elementFluxes.size() == elementFluxLabels.size());

    for (std::size_t dataIdx = 0; dataIdx < elementFluxes.size(); ++dataIdx) {
      std::vector<float> values(elementFluxes[dataIdx].size());
      for (std::size_t i = 0; i < elementFluxes[dataIdx].size(); ++i) {
        values[i] = static_cast<float>(elementFluxes[dataIdx][i]);
      }

      surfaceMesh_->getCellData().insertReplaceScalarData(
          std::move(values), elementFluxLabels[dataIdx]);
    }
  }

  static viennaray::TracingData<NumericType>
  movePointDataToRayData(viennals::PointData<NumericType> &pointData) {
    viennaray::TracingData<NumericType> rayData;

    const auto numData = pointData.getScalarDataSize();
    rayData.setNumberOfVectorData(numData);
    for (size_t i = 0; i < numData; ++i) {
      auto label = pointData.getScalarDataLabel(i);
      rayData.setVectorData(i, std::move(*pointData.getScalarData(label)),
                            label);
    }

    return std::move(rayData);
  }

private:
  viennaray::TraceTriangle<NumericType, D> rayTracer_;
  SmartPointer<ProcessModelCPU<NumericType, D>> model_;

  KDTreeType elementKdTree_;
  MeshType surfaceMesh_;
};

} // namespace viennaps
