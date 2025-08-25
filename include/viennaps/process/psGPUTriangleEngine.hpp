#pragma once

#ifndef VIENNACORE_COMPILE_GPU
#define VIENNACORE_COMPILE_GPU
#endif
#ifdef VIENNACORE_COMPILE_GPU

#include "../psDomain.hpp"
#include "psFluxEngine.hpp"
#include "psProcessBase.hpp"

#include <vcContext.hpp>

#include <lsAdvect.hpp>
#include <lsDomain.hpp>
#include <lsMesh.hpp>

#include <raygTrace.hpp>

#include <psgCreateSurfaceMesh.hpp>
#include <psgElementToPointData.hpp>
#include <psgPointToElementData.hpp>
#include <psgProcessModel.hpp>

namespace viennaps::gpu {

using namespace viennacore;

template <typename NumericType, int D>
class GPUTriangleEngine final : public FluxEngine<NumericType, D> {
  using TranslatorType = std::unordered_map<unsigned long, unsigned long>;
  using KDTreeType =
      SmartPointer<KDTree<NumericType, std::array<NumericType, 3>>>;
  using MeshType = SmartPointer<viennals::Mesh<float>>;

public:
  GPUTriangleEngine(Context &deviceContext)
      : deviceContext_(deviceContext), rayTracer_(deviceContext) {}

  ProcessResult checkInput(ProcessContext<NumericType, D> &context) final {

    auto model = std::dynamic_pointer_cast<gpu::ProcessModel<NumericType, D>>(
        context.model);
    if (!model) {
      Logger::getInstance().addWarning("Invalid GPU process model.").print();
      return ProcessResult::INVALID_INPUT;
    }

    const auto name = context.model->getProcessName().value_or("default");
    if (model->getParticleTypes().empty()) {
      Logger::getInstance()
          .addError("No particle types in process model: " + name)
          .print();
      return ProcessResult::INVALID_INPUT;
    }

    if (model->getPipelineFileName().empty()) {
      Logger::getInstance()
          .addError("No pipeline in process model: " + name)
          .print();
      return ProcessResult::INVALID_INPUT;
    }

    return ProcessResult::SUCCESS;
  }

  ProcessResult initialize(ProcessContext<NumericType, D> &context) final {
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

      auto model = std::dynamic_pointer_cast<gpu::ProcessModel<NumericType, D>>(
          context.model);
      rayTracer_.setPipeline(model->getPipelineFileName(),
                             deviceContext_.modulePath);
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
  }

  ProcessResult updateSurface(ProcessContext<NumericType, D> &context) final {
    surfaceMesh_ = viennals::Mesh<float>::New();
    if (!elementKdTree_)
      elementKdTree_ = KDTreeType::New();
    CreateSurfaceMesh<NumericType, float, D>(
        context.domain->getLevelSets().back(), surfaceMesh_, elementKdTree_)
        .apply();

    auto mesh =
        CreateTriangleMesh(context.domain->getGridDelta(), surfaceMesh_);
    rayTracer_.setGeometry(mesh);

    auto model = std::dynamic_pointer_cast<gpu::ProcessModel<NumericType, D>>(
        context.model);
    if (model->useMaterialIds()) {
      auto const &pointMaterialIds =
          *diskMesh_->getCellData().getScalarData("MaterialIds");
      std::vector<int> elementMaterialIds;
      auto &pointKdTree = translationField_->getKdTree();
      if (pointKdTree.getNumberOfPoints() != diskMesh_->nodes.size()) {
        pointKdTree.setPoints(diskMesh_->nodes);
        pointKdTree.build();
      }
      PointToElementDataSingle<NumericType, NumericType, int, float>(
          pointMaterialIds, elementMaterialIds, pointKdTree, surfaceMesh_)
          .apply();
      rayTracer_.setMaterialIds(elementMaterialIds);
    }

    assert(diskMesh_->nodes.size() > 0);
    assert(!surfaceMesh_->nodes.empty());

    return ProcessResult::SUCCESS;
  }

  ProcessResult
  calculateFluxes(ProcessContext<NumericType, D> &context,
                  viennacore::SmartPointer<viennals::PointData<NumericType>>
                      &fluxes) final {

    auto model = std::dynamic_pointer_cast<gpu::ProcessModel<NumericType, D>>(
        context.model);
    const auto name = model->getProcessName().value_or("default");
    const auto logLevel = Logger::getLogLevel();

    CudaBuffer d_coverages; // device buffer for coverages
    if (context.flags.useCoverages) {
      auto coverages = model->getSurfaceModel()->getCoverages();
      assert(coverages);
      assert(context.diskMesh);
      assert(translationField_);
      auto numCov = coverages->getScalarDataSize();
      auto &pointKdTree = translationField_->getKdTree();
      if (pointKdTree.getNumberOfPoints() != diskMesh_->nodes.size()) {
        pointKdTree.setPoints(diskMesh_->nodes);
        pointKdTree.build();
      }
      PointToElementData<NumericType, float>(d_coverages, coverages,
                                             pointKdTree, surfaceMesh_)
          .apply();
      rayTracer_.setElementData(d_coverages, numCov);
    }

    // run the ray tracer
    rayTracer_.apply();

    // extract fluxes on points
    ElementToPointData<NumericType, float>(
        rayTracer_.getResults(), fluxes, rayTracer_.getParticles(),
        elementKdTree_, diskMesh_, surfaceMesh_,
        domain_->getGridDelta() * context.rayTracingParams.smoothingNeighbors)
        .apply();

    // output
    if (logLevel >= static_cast<unsigned>(LogLevel::INTERMEDIATE)) {
      if (context.flags.useCoverages) {
        {
          auto coverages = model->getSurfaceModel()->getCoverages();
          downloadCoverages(d_coverages, surfaceMesh_->getCellData(), coverages,
                            surfaceMesh_->getElements<3>().size());
        }
        downloadResultsToPointData(surfaceMesh_->getCellData());
        static unsigned iterations = 0;
        viennals::VTKWriter<float>(surfaceMesh_,
                                   name + "_flux_" +
                                       std::to_string(iterations++) + ".vtp")
            .apply();
      }

      if (context.flags.useCoverages) {
        d_coverages.free();
      }
    }

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

    delete temp;
  }

  void downloadResultsToPointData(viennals::PointData<float> &pointData) {
    const auto numRates = rayTracer_.getNumberOfRates();
    const auto numPoints = rayTracer_.getNumberOfElements();
    assert(numRates > 0);
    assert(numPoints == surfaceMesh_->getElements<3>().size());
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

        pointData.insertReplaceScalarData(std::move(values), name);
      }
      offset += particles[pIdx].dataLabels.size();
    }
  }

private:
  Context &deviceContext_;
  viennaray::gpu::Trace<NumericType, D> rayTracer_;

  KDTreeType elementKdTree_;
  MeshType surfaceMesh_;

  bool rayTracerInitialized_ = false;
};

} // namespace viennaps::gpu

#endif
