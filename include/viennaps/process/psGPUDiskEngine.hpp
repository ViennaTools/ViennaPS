#pragma once

#ifdef VIENNACORE_COMPILE_GPU

#include "../psDomain.hpp"
#include "psFluxEngine.hpp"
#include "psProcessModel.hpp"

#include <vcContext.hpp>

#include <lsMesh.hpp>

#include <raygTrace.hpp>

namespace viennaps {

using namespace viennacore;

template <typename NumericType, int D>
class GPUDiskEngine final : public FluxEngine<NumericType, D> {
  using TranslatorType = std::unordered_map<unsigned long, unsigned long>;

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

    if (model->getPipelineFileName().empty()) {
      Logger::getInstance()
          .addWarning("No pipeline in process model: " + name)
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

      rayTracer_.setPipeline(model->getPipelineFileName(),
                             deviceContext_->modulePath);
      rayTracer_.setNumberOfRaysPerPoint(context.rayTracingParams.raysPerPoint);
      rayTracer_.setUseRandomSeeds(context.rayTracingParams.useRandomSeeds);
      rayTracer_.setPeriodicBoundary(periodicBoundary);
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

    auto points = diskMesh->getNodes();
    auto normals = *diskMesh->getCellData().getVectorData("Normals");
    auto materialIds = *diskMesh->getCellData().getScalarData("MaterialIds");

    // TODO: make this conversion to float prettier
    auto convertToFloat = [](std::vector<Vec3D<NumericType>> &input) {
      std::vector<Vec3Df> output;
      output.reserve(input.size());
      for (const auto &vec : input) {
        Vec3Df temp = {static_cast<float>(vec[0]), static_cast<float>(vec[1]),
                       static_cast<float>(vec[2])};
        output.emplace_back(temp);
      }
      return output;
    };

    std::vector<Vec3Df> fPoints = convertToFloat(points);
    std::vector<Vec3Df> fNormals = convertToFloat(normals);
    Vec3Df fMinExtent = {static_cast<float>(diskMesh->minimumExtent[0]),
                         static_cast<float>(diskMesh->minimumExtent[1]),
                         static_cast<float>(diskMesh->minimumExtent[2])};
    Vec3Df fMaxExtent = {static_cast<float>(diskMesh->maximumExtent[0]),
                         static_cast<float>(diskMesh->maximumExtent[1]),
                         static_cast<float>(diskMesh->maximumExtent[2])};

    viennaray::gpu::DiskMesh diskMeshRay{
        .points = fPoints,
        .normals = fNormals,
        .minimumExtent = fMinExtent,
        .maximumExtent = fMaxExtent,
        .radius = 0.f,
        .gridDelta = static_cast<float>(context.domain->getGridDelta())};

    if (context.rayTracingParams.diskRadius == 0.) {
      diskMeshRay.radius = static_cast<float>(context.domain->getGridDelta() *
                                              rayInternal::DiskFactor<D>);
    } else {
      diskMeshRay.radius =
          static_cast<float>(context.rayTracingParams.diskRadius);
    }

    rayTracer_.setGeometry(diskMeshRay);
    rayTracer_.setMaterialIds(materialIds);
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
      std::vector<float> cov(context.diskMesh->getNodes().size() * numCov, 0.f);

      for (int i = 0; i < numCov; ++i) {
        std::vector<NumericType> temp = *(coverages->getScalarData(i));
        std::vector<float> tempCasted(temp.size());
        std::transform(temp.begin(), temp.end(), tempCasted.begin(),
                       [](NumericType val) { return static_cast<float>(val); });
        assert(tempCasted.size() == context.diskMesh->getNodes().size());
        std::copy(tempCasted.begin(), tempCasted.end(),
                  cov.begin() + i * context.diskMesh->getNodes().size());
      }
      d_coverages.allocUpload(cov);

      rayTracer_.setElementData(d_coverages, numCov);
    }

    // run the ray tracer
    rayTracer_.apply();
    // TODO: smooth fluxes here
    downloadResultsToPointData(*fluxes);

    // output
    if (Logger::getLogLevel() >=
        static_cast<unsigned>(LogLevel::INTERMEDIATE)) {
      if (context.flags.useCoverages) {
        auto coverages = model->getSurfaceModel()->getCoverages();
        downloadCoverages(d_coverages, context.diskMesh->getCellData(),
                          coverages, context.diskMesh->getNodes().size());
      }
      downloadResultsToPointData(context.diskMesh->getCellData());
      static unsigned iterations = 0;
      viennals::VTKWriter<NumericType>(context.diskMesh,
                                 context.getProcessName() + "_flux_" +
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

  void downloadResultsToPointData(viennals::PointData<NumericType> &pointData) {
    const auto numRates = rayTracer_.getNumberOfRates();
    const auto numPoints = rayTracer_.getNumberOfElements();
    assert(numRates > 0);
    auto valueBuffer = rayTracer_.getResults();
    std::vector<float> tmpBuffer(numRates * numPoints);
    valueBuffer.download(tmpBuffer.data(), numPoints * numRates);
    auto particles = rayTracer_.getParticles();

    int offset = 0;
    for (int pIdx = 0; pIdx < particles.size(); pIdx++) {
      for (int dIdx = 0; dIdx < particles[pIdx].dataLabels.size(); dIdx++) {
        int tmpOffset = offset + dIdx;
        auto name = particles[pIdx].dataLabels[dIdx];

        std::vector<float> values(numPoints);
        std::memcpy(values.data(), &tmpBuffer[tmpOffset * numPoints],
                    numPoints * sizeof(float));

        std::vector<NumericType> diskFluxCasted(values.begin(), values.end());
        pointData.insertReplaceScalarData(std::move(diskFluxCasted), name);
      }
      offset += particles[pIdx].dataLabels.size();
    }
  }

private:
  std::shared_ptr<DeviceContext> deviceContext_;
  viennaray::gpu::Trace<NumericType, D> rayTracer_;

  bool rayTracerInitialized_ = false;
};

} // namespace viennaps

#endif
