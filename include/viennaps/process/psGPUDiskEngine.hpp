#pragma once

#ifdef VIENNACORE_COMPILE_GPU

#include "psDesorptionSource.hpp"
#include "psFluxEngine.hpp"
#include "psProcessModel.hpp"

#include <vcContext.hpp>

#include <gpu/raygTraceDisk.hpp>

namespace viennaps {

using namespace viennacore;

VIENNAPS_TEMPLATE_ND(NumericType, D)
class GPUDiskEngine final : public FluxEngine<NumericType, D> {
public:
  explicit GPUDiskEngine(std::shared_ptr<DeviceContext> deviceContext)
      : deviceContext_(deviceContext), rayTracer_(deviceContext) {}

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
    if (!rayTracerInitialized_) {
      rayTracer_.setParticleCallableMap(model_->getParticleCallableMap());
      rayTracer_.setCallables(model_->getCallableFileName(),
                              deviceContext_->modulePath);
      rayTracer_.setNumberOfRaysPerPoint(context.rayTracingParams.raysPerPoint);
      rayTracer_.setMaxBoundaryHits(context.rayTracingParams.maxBoundaryHits);
      if (context.rayTracingParams.maxReflections > 0)
        rayTracer_.setMaxReflections(context.rayTracingParams.maxReflections);
      rayTracer_.setUseRandomSeeds(context.rayTracingParams.useRandomSeeds);
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
    auto &diskMesh = context.diskMesh;
    assert(diskMesh != nullptr);

    const auto &[points, normals, materialIds] = context.getDiskMeshData();

    // TODO: make this conversion to float prettier
    auto convertToFloat = [](const std::vector<Vec3D<NumericType>> &input) {
      std::vector<Vec3Df> output(input.size());
#pragma omp parallel for
      for (std::int64_t i = 0; i < static_cast<std::int64_t>(input.size());
           ++i) {
        const auto &v = input[static_cast<std::size_t>(i)];
        output[static_cast<std::size_t>(i)] =
            Vec3Df{static_cast<float>(v[0]), static_cast<float>(v[1]),
                   static_cast<float>(v[2])};
      }
      return output;
    };

    Vec3Df fMinExtent = {static_cast<float>(diskMesh->minimumExtent[0]),
                         static_cast<float>(diskMesh->minimumExtent[1]),
                         static_cast<float>(diskMesh->minimumExtent[2])};
    Vec3Df fMaxExtent = {static_cast<float>(diskMesh->maximumExtent[0]),
                         static_cast<float>(diskMesh->maximumExtent[1]),
                         static_cast<float>(diskMesh->maximumExtent[2])};

    viennaray::DiskMesh diskMeshRay;
    diskMeshRay.nodes = convertToFloat(points);
    diskMeshRay.normals = convertToFloat(normals);
    diskMeshRay.minimumExtent = fMinExtent;
    diskMeshRay.maximumExtent = fMaxExtent;
    diskMeshRay.gridDelta = static_cast<float>(context.domain->getGridDelta());

    if (context.rayTracingParams.diskRadius == 0.) {
      diskMeshRay.radius = static_cast<float>(diskMeshRay.gridDelta *
                                              rayInternal::DiskFactor<D>);
    } else {
      diskMeshRay.radius =
          static_cast<float>(context.rayTracingParams.diskRadius);
    }
    diskRadius_ = diskMeshRay.radius;

    rayTracer_.setGeometry(diskMeshRay);

    if (model_->useMaterialIds()) {
      rayTracer_.setMaterialIds(materialIds);
    }
    assert(context.diskMesh->nodes.size() > 0);
    this->timer_.finish();

    return ProcessResult::SUCCESS;
  }

  ProcessResult calculateSourceFluxes(
      ProcessContext<NumericType, D> &context,
      SmartPointer<viennals::PointData<NumericType>> &fluxes) override {

    this->timer_.start();

    CudaBuffer d_coverages; // device buffer for coverages
    if (context.flags.useCoverages) {
      auto coverages = model_->getSurfaceModel()->getCoverages();
      assert(coverages);
      assert(context.diskMesh);
      auto numCov = coverages->getScalarDataSize();
      const auto numElements = context.diskMesh->getNodes().size();
      std::vector<float> cov(numElements * numCov, 0.f);

      for (int i = 0; i < numCov; ++i) {
        auto temp = coverages->getScalarData(i);
        std::transform(temp->begin(), temp->end(),
                       cov.begin() + i * numElements,
                       [](NumericType val) { return static_cast<float>(val); });
      }
      d_coverages.allocUpload(cov);

      rayTracer_.setElementData(d_coverages, numCov);
    }

    std::vector<NumericType> desorptionWeights;
    if (auto materialIds =
            context.diskMesh->getCellData().getScalarData("MaterialIds")) {
      desorptionWeights = model_->getSurfaceModel()
                              ->getDesorptionWeights(*materialIds)
                              .value_or(std::vector<NumericType>{});
    }

    // run the ray tracer
    rayTracer_.clearSurfaceSource();
    rayTracer_.apply();
    rayTracer_.normalizeResults();
    auto fluxResults = rayTracer_.getResults();
    copyResultsToPointData(*fluxes, context.rayTracingParams.smoothingNeighbors,
                           fluxResults);
    ++this->fluxCalculationsCount_;

    // output
    if (Logger::hasIntermediate()) {
      if (context.flags.useCoverages) {
        auto coverages = model_->getSurfaceModel()->getCoverages();
        downloadCoverages(d_coverages, context.diskMesh->getCellData(),
                          coverages, context.diskMesh->getNodes().size());
      }
      copyResultsToPointData(context.diskMesh->getCellData(),
                             context.rayTracingParams.smoothingNeighbors,
                             fluxResults);
      viennals::VTKWriter<NumericType>(
          context.diskMesh,
          context.intermediateOutputPath + context.getProcessName() + "_flux_" +
              std::to_string(context.currentIteration) + ".vtp")
          .apply();
    }

    d_coverages.free();
    this->timer_.finish();

    return ProcessResult::SUCCESS;
  }

  ProcessResult calculateSurfaceFluxes(
      ProcessContext<NumericType, D> &context,
      SmartPointer<viennals::PointData<NumericType>> &fluxes) override {

    this->timer_.start();

    const auto &[nodes, normals, materialIds] = context.getDiskMeshData();
    const auto desorptionWeights = model_->getSurfaceModel()
                                       ->getDesorptionWeights(materialIds)
                                       .value_or(std::vector<NumericType>{});

    if (desorptionWeights.size() != nodes.size()) {
      VIENNACORE_LOG_WARNING(
          "Desorption weights size does not match number of mesh nodes. "
          "Skipping surface flux calculation.");
      return ProcessResult::INVALID_INPUT;
    }

    auto sourceData = makeDiskDesorptionSourceData<float, NumericType, D>(
        nodes, normals, desorptionWeights, context.domain->getGridDelta(),
        static_cast<NumericType>(diskRadius_), true);
    if (!sourceData.hasSource) {
      // No active desorption sources, skip ray tracing
      VIENNACORE_LOG_DEBUG(
          "No active desorption sources found. Skipping ray tracing.");
      return ProcessResult::INVALID_INPUT;
    }

    rayTracer_.setSurfaceSource(sourceData.positions, sourceData.normals,
                                sourceData.weights, sourceData.sourceArea,
                                sourceData.sourceOffset);
    rayTracer_.apply();
    rayTracer_.normalizeResults();
    auto desorptionResults = rayTracer_.getResults();

    copyResultsToPointData(*fluxes, context.rayTracingParams.smoothingNeighbors,
                           desorptionResults);

    rayTracer_.clearSurfaceSource();
    this->timer_.finish();

    return ProcessResult::SUCCESS;
  }

private:
  static void
  downloadCoverages(CudaBuffer &d_coverages,
                    viennals::PointData<NumericType> &elementData,
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

      std::vector<NumericType> valuesCasted(values.begin(), values.end());
      elementData.insertReplaceScalarData(std::move(valuesCasted), covName);
    }

    delete[] temp;
  }

  void copyResultsToPointData(
      viennals::PointData<NumericType> &pointData, int smoothingNeighbors,
      std::vector<std::vector<viennaray::gpu::ResultType>> results) {
    const auto numRates = rayTracer_.getNumberOfRates();
    const auto numPoints = rayTracer_.getNumberOfElements();
    assert(numRates > 0);
    auto particles = rayTracer_.getParticles();

    int offset = 0;
    for (int pIdx = 0; pIdx < particles.size(); pIdx++) {
      for (int dIdx = 0; dIdx < particles[pIdx].dataLabels.size(); dIdx++) {
        auto diskFlux = results[offset + dIdx];
        if (smoothingNeighbors > 0) {
          rayTracer_.smoothFlux(diskFlux, smoothingNeighbors);
        }
        auto name = particles[pIdx].dataLabels[dIdx];

        std::vector<NumericType> diskFluxCasted(diskFlux.begin(),
                                                diskFlux.end());
        pointData.insertReplaceScalarData(std::move(diskFluxCasted), name);
      }
      offset += particles[pIdx].dataLabels.size();
    }
  }

private:
  std::shared_ptr<DeviceContext> deviceContext_;
  viennaray::gpu::TraceDisk<NumericType, D> rayTracer_;
  SmartPointer<gpu::ProcessModelGPU<NumericType, D>> model_;

  bool rayTracerInitialized_ = false;
  float diskRadius_ = 0.f;
};

} // namespace viennaps

#endif
