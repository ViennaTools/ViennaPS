#pragma once

#include "../psCreateSurfaceMesh.hpp"
#include "../psElementToPointData.hpp"
#include "../psPointToElementData.hpp"

#include "psDesorptionSource.hpp"
#include "psFluxEngine.hpp"
#include "psProcessModel.hpp"

#include <rayTraceTriangle.hpp>

#include <cmath>    // TEST (LS_IONNORM)
#include <cstdlib>  // TEST (LS_IONNORM)
#include <iostream> // TEST (LS_IONNORM)

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

    surfaceMesh_ = MeshType::New();
    elementKdTree_ = KDTreeType::New();

    postProcessing_.setDataLabels(model_->getParticleDataLabels());
    postProcessing_.setConversionRadius(
        context.domain->getGridDelta() *
        (context.rayTracingParams.smoothingNeighbors + 1));
    postProcessing_.setSurfaceMesh(surfaceMesh_);
    postProcessing_.setElementKdTree(elementKdTree_);
    postProcessing_.setDiskMesh(context.diskMesh);

    return ProcessResult::SUCCESS;
  }

  ProcessResult
  updateSurface(ProcessContext<NumericType, D> &context) override {
    this->timer_.start();
    assert(surfaceMesh_ != nullptr);
    assert(elementKdTree_ != nullptr);

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

      // TEST (LS_IONNORM): element normals averaged, area-weighted, over the
      // elements within LS_IONNORM grid spacings, for the ion's incidence
      // angle (the cell set's plane fit over ionNormR_ cells)
      ionnorm::averaged.clear();
      if (const char *e = std::getenv("LS_IONNORM"); e && std::atof(e) > 0) {
        const NumericType radius =
            std::atof(e) * context.domain->getGridDelta();
        const auto &nodes = surfaceMesh_->nodes;
        const auto &tris = surfaceMesh_->triangles;
        const auto &nrm = *surfaceMesh_->getNormals();
        const auto numTris = tris.size();
        std::vector<double> area(numTris);
        std::vector<std::array<NumericType, 3>> centers(numTris);
        for (size_t i = 0; i < numTris; ++i) {
          const auto &a = nodes[tris[i][0]], &b = nodes[tris[i][1]],
                     &c = nodes[tris[i][2]];
          area[i] = 0.5 * Norm(CrossProduct(b - a, c - a));
          for (int k = 0; k < 3; ++k)
            centers[i][k] = (a[k] + b[k] + c[k]) / NumericType(3);
        }
        ionnorm::averaged.resize(numTris);
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < numTris; ++i) {
          const auto close =
              elementKdTree_->findNearestWithinRadius(centers[i], radius)
                  .value();
          std::array<double, 3> s{0., 0., 0.};
          double w = 0.;
          for (const auto &[j, dist] : close) {
            if (DotProduct(nrm[i], nrm[j]) < -0.5f)
              continue; // the far side of a thin feature
            for (int k = 0; k < 3; ++k)
              s[k] += area[j] * nrm[j][k];
            w += area[j];
          }
          const double l = std::sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
          if (w > 0. && l > 0.5 * w)
            ionnorm::averaged[i] = {float(s[0] / l), float(s[1] / l),
                                    float(s[2] / l)};
          else
            ionnorm::averaged[i] = nrm[i];
        }
        ionnormElems_ = numTris;
        ionnormTurned_ = 0;
        for (size_t i = 0; i < numTris; ++i)
          if (DotProduct(ionnorm::averaged[i], nrm[i]) < 0.9397f) // 20 deg
            ++ionnormTurned_;
        if (!ionnormAnnounced_) {
          ionnormAnnounced_ = true;
          std::cout << "  [level set: ion incidence normal averaged within "
                    << std::atof(e) << " grid spacings (" << radius
                    << " nm)]\n";
        }
      }
    }

    auto const &pointMaterialIds = *context.diskMesh->getMaterialIds();
    std::vector<int> elementMaterialIds;
    auto pointKdTree = context.getPointKdTree();
    PointToElementDataSingle<NumericType, NumericType, int, float>(
        pointMaterialIds, elementMaterialIds, *pointKdTree, surfaceMesh_)
        .apply();
    rayTracer_.setMaterialIds(elementMaterialIds);

    assert(context.diskMesh->nodes.size() > 0);
    assert(!surfaceMesh_->nodes.empty());
    this->timer_.finish();

    return ProcessResult::SUCCESS;
  }

  ProcessResult
  calculateSourceFluxes(ProcessContext<NumericType, D> &context,
                        SmartPointer<PointData<NumericType>> &fluxes) override {
    this->timer_.start();
    auto surfaceModel = context.model->getSurfaceModel();

    // set coverages in ray tracer
    if (context.flags.useCoverages) {
      auto globalTracingData = PointData<NumericType>::New();
      auto pointKdTree = context.getPointKdTree();
      PointToElementData<NumericType, float>(
          *globalTracingData, surfaceModel->getCoverages(), *pointKdTree,
          surfaceMesh_, Logger::hasIntermediate())
          .apply();
      rayTracer_.setGlobalData(globalTracingData);
    }

    auto elementFluxes = runRayTracer(context);

    if (!ionnorm::averaged.empty() && ++ionnormCalls_ % 200 == 0)
      std::cout << "  [ion normal: " << ionnorm::hits << " ion hits, "
                << ionnorm::facet << " from behind the averaged surface, "
                << ionnorm::stopped
                << " reflections stopped by the struck facet; elements "
                   "more than 20 deg from their averaged normal: "
                << ionnormTurned_ << " of " << ionnormElems_ << "]\n";

    // post-process fluxes
    postProcessing_.setPointData(fluxes);
    postProcessing_.setElementDataArrays(std::move(elementFluxes));
    postProcessing_.apply();

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
      ProcessContext<NumericType, D> &context,
      SmartPointer<PointData<NumericType>> &fluxes) override {

    this->timer_.start();

    const auto &[nodes, normals, materialIds] = context.getDiskMeshData();
    auto desorptionWeights = model_->getSurfaceModel()
                                 ->getDesorptionWeights(materialIds)
                                 .value_or(std::vector<NumericType>{});

    if (desorptionWeights.size() != nodes.size()) {
      VIENNACORE_LOG_WARNING(
          "Desorption weights size does not match number of mesh nodes. "
          "Skipping surface flux calculation.");
      return ProcessResult::INVALID_INPUT;
    }

    // Map desorption weights from disk mesh nodes to surface mesh elements
    auto pointKdTree = context.getPointKdTree();
    assert(surfaceMesh_ && "Surface mesh not initialized.");
    std::vector<NumericType> elementWeights;
    PointToElementDataSingle<NumericType, NumericType, NumericType, float>(
        desorptionWeights, elementWeights, *pointKdTree, surfaceMesh_)
        .apply();

    const auto &triangles = surfaceMesh_->triangles;
    const auto &triangleNodes = surfaceMesh_->nodes;
    const auto triangleNormals = surfaceMesh_->getNormals();
    assert(triangleNormals);

    auto sourceData = makeTriangleDesorptionSourceData<NumericType>(
        triangleNodes, triangles, *triangleNormals, elementWeights,
        static_cast<NumericType>(context.domain->getGridDelta()));
    if (!sourceData.hasSource) {
      // No active desorption sources, skip ray tracing
      VIENNACORE_LOG_DEBUG(
          "No active desorption sources found. Skipping ray tracing.");
      return ProcessResult::INVALID_INPUT;
    }

    auto source = std::make_shared<DesorptionSource<NumericType, D>>(
        std::move(sourceData), context.rayTracingParams.raysPerPoint);

    rayTracer_.setSource(source);
    auto elementFluxes = runRayTracer(context);

    // post-process fluxes
    fluxes->clear();
    postProcessing_.setPointData(fluxes);
    postProcessing_.setElementDataArrays(std::move(elementFluxes));
    postProcessing_.setPostFix("_surface");
    postProcessing_.prepare(true); // prepare without building the tree again
    postProcessing_.validate();
    postProcessing_.convert();
    postProcessing_.setPostFix(""); // clear postfix

    // reset source
    if (auto modelSource = model_->getSource()) {
      rayTracer_.setSource(modelSource);
    } else {
      rayTracer_.resetSource();
    }

    this->timer_.finish();

    return ProcessResult::SUCCESS;
  }

private:
  auto runRayTracer(ProcessContext<NumericType, D> &context) {
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
        auto flux = std::move(*localData.getScalarData(i));

        // normalize
        rayTracer_.normalizeFlux(flux,
                                 context.rayTracingParams.normalizationType);

        particleFluxLabels.push_back(localData.getScalarDataLabel(i));
        particleFluxes.push_back(std::move(flux));
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

    return elementFluxes;
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

private:
  viennaray::TraceTriangle<NumericType, D> rayTracer_;
  SmartPointer<ProcessModelCPU<NumericType, D>> model_;
  PostProcessingType postProcessing_;

  KDTreeType elementKdTree_;
  MeshType surfaceMesh_;
  bool ionnormAnnounced_ = false;  // TEST (LS_IONNORM)
  unsigned long ionnormCalls_ = 0; // TEST (LS_IONNORM)
  size_t ionnormElems_ = 0, ionnormTurned_ = 0; // TEST (LS_IONNORM)
};

} // namespace viennaps
