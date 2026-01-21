#pragma once

#include "psFluxEngine.hpp"
#include "psProcessModel.hpp"

#include <rayTraceDisk.hpp>

namespace viennaps {

using namespace viennacore;

VIENNAPS_TEMPLATE_ND(NumericType, D)
class CPUDiskEngine final : public FluxEngine<NumericType, D> {
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
    auto &diskMesh = context.diskMesh;
    assert(diskMesh != nullptr);
    assert(model_ != nullptr);

    auto points = diskMesh->getNodes();
    auto normals = *diskMesh->getCellData().getVectorData("Normals");
    auto materialIds = *diskMesh->getCellData().getScalarData("MaterialIds");

    if (context.rayTracingParams.diskRadius == 0.) {
      rayTracer_.setGeometry(points, normals,
                             context.domain->getGrid().getGridDelta());
    } else {
      rayTracer_.setGeometry(points, normals,
                             context.domain->getGrid().getGridDelta(),
                             context.rayTracingParams.diskRadius);
    }
    rayTracer_.setMaterialIds(materialIds);
    this->timer_.finish();

    return ProcessResult::SUCCESS;
  }

  ProcessResult calculateFluxes(
      ProcessContext<NumericType, D> &context,
      SmartPointer<viennals::PointData<NumericType>> &fluxes) override {
    assert(model_ != nullptr);
    this->timer_.start();
    viennaray::TracingData<NumericType> rayTracingData;
    auto surfaceModel = context.model->getSurfaceModel();

    // move coverages to the ray tracer
    if (context.flags.useCoverages) {
      rayTracingData = movePointDataToRayData(surfaceModel->getCoverages());
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

    // move coverages back in the model
    if (context.flags.useCoverages)
      moveRayDataToPointData(surfaceModel->getCoverages(), rayTracingData);
    this->timer_.finish();

    return ProcessResult::SUCCESS;
  }

private:
  void runRayTracer(ProcessContext<NumericType, D> const &context,
                    SmartPointer<viennals::PointData<NumericType>> &fluxes) {
    assert(fluxes != nullptr);
    assert(model_ != nullptr);
    fluxes->clear();

    unsigned particleIdx = 0;
    for (auto &particle : model_->getParticleTypes()) {
      int dataLogSize = model_->getParticleLogSize(particleIdx);
      if (dataLogSize > 0) {
        rayTracer_.getDataLog().data.resize(1);
        rayTracer_.getDataLog().data[0].resize(dataLogSize, 0.);
      }
      rayTracer_.setParticleType(particle);
      rayTracer_.apply();

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
      for (int i = 0; i < numFluxes; ++i) {
        auto &flux = localData.getVectorData(i);

        // normalize and smooth
        rayTracer_.normalizeFlux(flux,
                                 context.rayTracingParams.normalizationType);
        rayTracer_.smoothFlux(flux,
                              context.rayTracingParams.smoothingNeighbors);

        fluxes->insertNextScalarData(std::move(flux),
                                     localData.getVectorDataLabel(i));
      }

      model_->mergeParticleData(rayTracer_.getDataLog(), particleIdx);
      ++particleIdx;
    }
  }

  static viennaray::TracingData<NumericType> movePointDataToRayData(
      SmartPointer<viennals::PointData<NumericType>> pointData) {
    viennaray::TracingData<NumericType> rayData;
    const auto numData = pointData->getScalarDataSize();
    rayData.setNumberOfVectorData(numData);
    for (size_t i = 0; i < numData; ++i) {
      auto label = pointData->getScalarDataLabel(i);
      rayData.setVectorData(i, std::move(*pointData->getScalarData(label)),
                            label);
    }
    return rayData;
  }

  static void moveRayDataToPointData(
      SmartPointer<viennals::PointData<NumericType>> pointData,
      viennaray::TracingData<NumericType> &rayData) {
    pointData->clear();
    const auto numData = rayData.getVectorData().size();
    for (size_t i = 0; i < numData; ++i)
      pointData->insertNextScalarData(std::move(rayData.getVectorData(i)),
                                      rayData.getVectorDataLabel(i));
  }

private:
  viennaray::TraceDisk<NumericType, D> rayTracer_;
  SmartPointer<ProcessModelCPU<NumericType, D>> model_;
};

} // namespace viennaps