#pragma once

#ifdef VIENNACORE_COMPILE_GPU

#include "../psDomain.hpp"
#include "psFluxEngine.hpp"
#include "psProcessModel.hpp"

#include <vcContext.hpp>

#include <lsMesh.hpp>

#include <raygTraceDisk.hpp>

namespace viennaps {

using namespace viennacore;

template <typename NumericType, int D>
class GPUDiskEngine final : public FluxEngine<NumericType, D> {
public:
  GPUDiskEngine(std::shared_ptr<DeviceContext> deviceContext)
      : deviceContext_(deviceContext), rayTracer_(deviceContext) {}

  ProcessResult checkInput(ProcessContext<NumericType, D> &context) final {

    auto model =
        std::dynamic_pointer_cast<gpu::ProcessModelGPU<NumericType, D>>(
            context.model);
    if (!model) {
      Logger::getInstance().addWarning("Invalid GPU process model.").print();
      return ProcessResult::INVALID_INPUT;
    }

    const auto name = context.model->getProcessName().value_or("default");
    if (model->getParticleTypes().empty()) {
      Logger::getInstance()
          .addWarning("No particle types in process model: " + name)
          .print();
      return ProcessResult::INVALID_INPUT;
    }

    if (model->getCallableFileName().empty()) {
      Logger::getInstance()
          .addWarning("No callables in process model: " + name)
          .print();
      return ProcessResult::INVALID_INPUT;
    }

    return ProcessResult::SUCCESS;
  }

  ProcessResult initialize(ProcessContext<NumericType, D> &context) final {
    auto model =
        std::dynamic_pointer_cast<gpu::ProcessModelGPU<NumericType, D>>(
            context.model);
    if (!rayTracerInitialized_) {
      // Check for periodic boundary conditions
      bool periodicBoundary = false;
      if (context.rayTracingParams.ignoreFluxBoundaries) {
        Logger::getInstance()
            .addWarning("Ignoring flux boundaries not implemented on GPU.")
            .print();
      } else {
        const auto &grid = context.domain->getGrid();
        for (unsigned i = 0; i < D; ++i) {
          if (grid.getBoundaryConditions(i) ==
              viennals::BoundaryConditionEnum::PERIODIC_BOUNDARY) {
            periodicBoundary = true;
            break;
          }
        }
      }

      rayTracer_.setParticleCallableMap(model->getParticleCallableMap());
      rayTracer_.setCallables(model->getCallableFileName(),
                              deviceContext_->modulePath);
      rayTracer_.setNumberOfRaysPerPoint(context.rayTracingParams.raysPerPoint);
      rayTracer_.setMaxBoundaryHits(context.rayTracingParams.maxBoundaryHits);
      if (context.rayTracingParams.maxReflections > 0)
        rayTracer_.setMaxReflections(context.rayTracingParams.maxReflections);
      rayTracer_.setUseRandomSeeds(context.rayTracingParams.useRandomSeeds);
      if (!context.rayTracingParams.useRandomSeeds)
        rayTracer_.setRngSeed(context.rayTracingParams.rngSeed);
      rayTracer_.setPeriodicBoundary(periodicBoundary);
      rayTracer_.setIgnoreBoundary(
          context.rayTracingParams.ignoreFluxBoundaries);
      for (auto &particle : model->getParticleTypes()) {
        rayTracer_.insertNextParticle(particle);
      }
      rayTracer_.prepareParticlePrograms();
    }
    rayTracer_.setParameters(model->getProcessDataDPtr());
    rayTracerInitialized_ = true;

    return ProcessResult::SUCCESS;
  }

  ProcessResult updateSurface(ProcessContext<NumericType, D> &context) final {
    this->timer_.start();
    auto &diskMesh = context.diskMesh;
    assert(diskMesh != nullptr);

    auto const &points = diskMesh->getNodes();
    auto const &normals = *diskMesh->getCellData().getVectorData("Normals");

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

    rayTracer_.setGeometry(diskMeshRay);

    auto model =
        std::dynamic_pointer_cast<gpu::ProcessModelGPU<NumericType, D>>(
            context.model);
    if (model->useMaterialIds()) {
      auto const &materialIds =
          *diskMesh->getCellData().getScalarData("MaterialIds");
      rayTracer_.setMaterialIds(materialIds);
    }
    assert(context.diskMesh->nodes.size() > 0);
    this->timer_.finish();

    return ProcessResult::SUCCESS;
  }

  ProcessResult
  calculateFluxes(ProcessContext<NumericType, D> &context,
                  viennacore::SmartPointer<viennals::PointData<NumericType>>
                      &fluxes) final {

    this->timer_.start();
    auto model =
        std::dynamic_pointer_cast<gpu::ProcessModelGPU<NumericType, D>>(
            context.model);

    CudaBuffer d_coverages; // device buffer for coverages
    if (context.flags.useCoverages) {
      auto coverages = model->getSurfaceModel()->getCoverages();
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

    // run the ray tracer
    rayTracer_.apply();
    rayTracer_.normalizeResults();
    downloadResultsToPointData(*fluxes,
                               context.rayTracingParams.smoothingNeighbors);

    // output
    if (Logger::getLogLevel() >=
        static_cast<unsigned>(LogLevel::INTERMEDIATE)) {
      if (context.flags.useCoverages) {
        auto coverages = model->getSurfaceModel()->getCoverages();
        downloadCoverages(d_coverages, context.diskMesh->getCellData(),
                          coverages, context.diskMesh->getNodes().size());
      }
      downloadResultsToPointData(context.diskMesh->getCellData(),
                                 context.rayTracingParams.smoothingNeighbors);
      static unsigned iterations = 0;
      viennals::VTKWriter<NumericType>(
          context.diskMesh, context.getProcessName() + "_flux_" +
                                std::to_string(iterations++) + ".vtp")
          .apply();

      if (context.flags.useCoverages) {
        d_coverages.free();
      }
    }
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

    delete temp;
  }

  void downloadResultsToPointData(viennals::PointData<NumericType> &pointData,
                                  int smoothingNeighbors) {
    const auto numRates = rayTracer_.getNumberOfRates();
    const auto numPoints = rayTracer_.getNumberOfElements();
    assert(numRates > 0);
    auto particles = rayTracer_.getParticles();

    int offset = 0;
    for (int pIdx = 0; pIdx < particles.size(); pIdx++) {
      for (int dIdx = 0; dIdx < particles[pIdx].dataLabels.size(); dIdx++) {
        auto diskFlux = rayTracer_.getFlux(pIdx, dIdx, smoothingNeighbors);
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

  bool rayTracerInitialized_ = false;
};

} // namespace viennaps

#endif
