#pragma once

#include "../psCreateSurfaceMesh.hpp"
#include "../psElementToPointData.hpp"
#include "../psPointToElementData.hpp"

#include "psFluxEngine.hpp"
#include "psProcessModel.hpp"

#include <rayTraceTriangle.hpp>

namespace viennaps {

using namespace viennacore;

/// This class server as the main process tool, applying a user- or pre-defined
/// process model to a domain. Depending on the user inputs surface advection, a
/// single callback function or a geometric advection is applied.
template <typename NumericType, int D>
class CPUTriangleEngine final : public FluxEngine<NumericType, D> {
  using KDTreeType =
      SmartPointer<KDTree<NumericType, std::array<NumericType, 3>>>;
  using MeshType = SmartPointer<viennals::Mesh<NumericType>>;

public:
  ProcessResult checkInput(ProcessContext<NumericType, D> &context) final {
    auto model = std::dynamic_pointer_cast<ProcessModelCPU<NumericType, D>>(
        context.model);
    if (!model) {
      Logger::getInstance().addWarning("Invalid process model.").print();
      return ProcessResult::INVALID_INPUT;
    }
    return ProcessResult::SUCCESS;
  }

  ProcessResult initialize(ProcessContext<NumericType, D> &context) final {
    // Map the domain boundary to the ray tracing boundaries
    viennaray::BoundaryCondition rayBoundaryCondition[D];
    if (context.rayTracingParams.ignoreFluxBoundaries) {
      for (unsigned i = 0; i < D; ++i)
        rayBoundaryCondition[i] = viennaray::BoundaryCondition::IGNORE;
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
    rayTracer_.setUseRandomSeeds(context.rayTracingParams.useRandomSeeds);
    if (!context.rayTracingParams.useRandomSeeds)
      rayTracer_.setRngSeed(context.rayTracingParams.rngSeed);

    auto model = std::dynamic_pointer_cast<ProcessModelCPU<NumericType, D>>(
        context.model);

    if (auto source = model->getSource()) {
      rayTracer_.setSource(source);
      Logger::getInstance().addInfo("Using custom source.").print();
    }
    if (auto primaryDirection = model->getPrimaryDirection()) {
      Logger::getInstance()
          .addInfo("Using primary direction: " +
                   util::arrayToString(primaryDirection.value()))
          .print();
      rayTracer_.setPrimaryDirection(primaryDirection.value());
    }

    model->initializeParticleDataLogs();

    return ProcessResult::SUCCESS;
  }

  ProcessResult updateSurface(ProcessContext<NumericType, D> &context) final {
    this->timer_.start();
    surfaceMesh_ = viennals::Mesh<NumericType>::New();
    if (!elementKdTree_)
      elementKdTree_ = KDTreeType::New();
    CreateSurfaceMesh<NumericType, NumericType, D>(
        context.domain->getLevelSets().back(), surfaceMesh_, elementKdTree_,
        1e-12, context.rayTracingParams.minNodeDistanceFactor)
        .apply();

    rayTracer_.setGeometry(surfaceMesh_->nodes, surfaceMesh_->triangles,
                           context.domain->getGridDelta());

    auto const &pointMaterialIds =
        *context.diskMesh->getCellData().getScalarData("MaterialIds");
    std::vector<int> elementMaterialIds;
    auto &pointKdTree = context.translationField->getKdTree();
    if (pointKdTree->getNumberOfPoints() != context.diskMesh->nodes.size()) {
      pointKdTree->setPoints(context.diskMesh->nodes);
      pointKdTree->build();
    }
    PointToElementDataSingle<NumericType, NumericType, int, NumericType>(
        pointMaterialIds, elementMaterialIds, *pointKdTree, surfaceMesh_)
        .apply();
    rayTracer_.setMaterialIds(elementMaterialIds);

    assert(context.diskMesh->nodes.size() > 0);
    assert(!surfaceMesh_->nodes.empty());
    this->timer_.finish();

    return ProcessResult::SUCCESS;
  }

  ProcessResult
  calculateFluxes(ProcessContext<NumericType, D> &context,
                  viennacore::SmartPointer<viennals::PointData<NumericType>>
                      &fluxes) final {
    this->timer_.start();
    viennaray::TracingData<NumericType> rayTracingData;
    auto surfaceModel = context.model->getSurfaceModel();

    // copy coverages to the ray tracer
    if (context.flags.useCoverages) {
      auto &pointKdTree = context.translationField->getKdTree();
      if (pointKdTree->getNumberOfPoints() != context.diskMesh->nodes.size()) {
        pointKdTree->setPoints(context.diskMesh->nodes);
        pointKdTree->build();
      }
      // Coverages are copied to elementData so there is no need to move them
      // back to the model
      viennals::PointData<NumericType> elementData;
      PointToElementData<NumericType, NumericType>(
          elementData, surfaceModel->getCoverages(), *pointKdTree, surfaceMesh_,
          Logger::getLogLevel() >=
              static_cast<unsigned>(LogLevel::INTERMEDIATE))
          .apply();
      // Move data from PointData to TracingData
      rayTracingData = MovePointDataToRayData(elementData);
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

    runRayTracer(context, fluxes);

    // output
    if (Logger::getLogLevel() >=
        static_cast<unsigned>(LogLevel::INTERMEDIATE)) {
      static unsigned iterations = 0;
      viennals::VTKWriter<NumericType>(
          surfaceMesh_, context.getProcessName() + "_flux_" +
                            std::to_string(iterations++) + ".vtp")
          .apply();
    }

    this->timer_.finish();

    return ProcessResult::SUCCESS;
  }

private:
  void runRayTracer(ProcessContext<NumericType, D> const &context,
                    SmartPointer<viennals::PointData<NumericType>> &fluxes) {
    assert(fluxes != nullptr);
    fluxes->clear();

    auto model = std::dynamic_pointer_cast<ProcessModelCPU<NumericType, D>>(
        context.model);

    std::vector<std::vector<NumericType>> elementFluxes;

    unsigned particleIdx = 0;
    for (auto &particle : model->getParticleTypes()) {
      int dataLogSize = model->getParticleLogSize(particleIdx);
      if (dataLogSize > 0) {
        rayTracer_.getDataLog().data.resize(1);
        rayTracer_.getDataLog().data[0].resize(dataLogSize, 0.);
      }
      rayTracer_.setParticleType(particle);
      rayTracer_.apply();

      auto info = rayTracer_.getRayTraceInfo();

      if (Logger::getLogLevel() >= 5) {
        Logger::getInstance()
            .addDebug(
                "Particle " + std::to_string(particleIdx) +
                "\n\tRays Traced: " + std::to_string(info.totalRaysTraced) +
                "\n\tNon-Geometry Hits: " +
                std::to_string(info.nonGeometryHits) +
                "\n\tGeometry Hits: " + std::to_string(info.geometryHits) +
                "\n\tParticle Hits: " + std::to_string(info.particleHits) +
                (info.warning
                     ? "\n\tWarning during ray tracing."
                     : (info.error ? "\n\tError during ray tracing." : "")))
            .print();
      }

      // fill up fluxes vector with fluxes from this particle type
      auto &localData = rayTracer_.getLocalData();
      int numFluxes = particle->getLocalDataLabels().size();
      for (int i = 0; i < numFluxes; ++i) {
        auto flux = std::move(localData.getVectorData(i));

        // normalize
        rayTracer_.normalizeFlux(flux,
                                 context.rayTracingParams.normalizationType);

        // output
        if (Logger::getLogLevel() >=
            static_cast<unsigned>(LogLevel::INTERMEDIATE)) {
          surfaceMesh_->getCellData().insertReplaceScalarData(
              flux, particle->getLocalDataLabels()[i]);
        }

        elementFluxes.push_back(std::move(flux));
      }

      model->mergeParticleData(rayTracer_.getDataLog(), particleIdx);
      ++particleIdx;
    }

    // map fluxes on points
    ElementToPointData<NumericType, NumericType>(
        elementFluxes, fluxes, model->getParticleTypes(), elementKdTree_,
        context.diskMesh, surfaceMesh_,
        context.domain->getGridDelta() *
            (context.rayTracingParams.smoothingNeighbors + 1))
        .apply();
  }

  static viennaray::TracingData<NumericType>
  MovePointDataToRayData(viennals::PointData<NumericType> &pointData) {
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

  KDTreeType elementKdTree_;
  MeshType surfaceMesh_;
};

} // namespace viennaps