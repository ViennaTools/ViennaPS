#pragma once

#ifdef VIENNACORE_COMPILE_GPU

#include "../psDomain.hpp"
#include "psFluxEngine.hpp"
#include "psProcessModel.hpp"

#include <vcContext.hpp>

#include <lsMesh.hpp>

#include <raygTrace.hpp>
#include <raygTraceLine.hpp>

namespace viennaps {

using namespace viennacore;

template <typename NumericType, int D>
class GPULineEngine final : public FluxEngine<NumericType, D> {
  using KDTreeType =
      SmartPointer<KDTree<NumericType, std::array<NumericType, 3>>>;
  using MeshType = SmartPointer<viennals::Mesh<NumericType>>;

public:
  GPULineEngine(std::shared_ptr<DeviceContext> deviceContext)
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

    surfaceMesh_ = viennals::Mesh<NumericType>::New();
    if (!elementKdTree_) {
      elementKdTree_ = KDTreeType::New();
    }
    viennals::ToSurfaceMesh<NumericType, D>(context.domain->getSurface(),
                                            surfaceMesh_)
        .apply();

    std::vector<Vec3D<NumericType>> elementCenters(surfaceMesh_->lines.size());
    std::vector<Vec3Df> fLineNormals(surfaceMesh_->lines.size());
    for (int i = 0; i < surfaceMesh_->lines.size(); ++i) {
      Vec3D<NumericType> p0 = {
          surfaceMesh_->nodes[surfaceMesh_->lines[i][0]][0],
          surfaceMesh_->nodes[surfaceMesh_->lines[i][0]][1],
          surfaceMesh_->nodes[surfaceMesh_->lines[i][0]][2]};
      Vec3D<NumericType> p1 = {
          surfaceMesh_->nodes[surfaceMesh_->lines[i][1]][0],
          surfaceMesh_->nodes[surfaceMesh_->lines[i][1]][1],
          surfaceMesh_->nodes[surfaceMesh_->lines[i][1]][2]};
      elementCenters[i] = (p0 + p1) / NumericType(2);

      Vec3D<NumericType> lineDir = p1 - p0;
      Vec3D<NumericType> normal =
          Vec3D<NumericType>{-lineDir[1], lineDir[0], 0};
      Normalize(normal);
      fLineNormals[i] =
          Vec3Df{static_cast<float>(normal[0]), static_cast<float>(normal[1]),
                 static_cast<float>(normal[2])};
    }
    elementKdTree_->setPoints(elementCenters);
    elementKdTree_->build();

    std::vector<Vec3Df> fNodes(surfaceMesh_->nodes.size());
    for (size_t i = 0; i < surfaceMesh_->nodes.size(); i++) {
      fNodes[i] = {static_cast<float>(surfaceMesh_->nodes[i][0]),
                   static_cast<float>(surfaceMesh_->nodes[i][1]),
                   static_cast<float>(surfaceMesh_->nodes[i][2])};
    }
    Vec3Df fMinExtent = {static_cast<float>(diskMesh->minimumExtent[0]),
                         static_cast<float>(diskMesh->minimumExtent[1]),
                         static_cast<float>(diskMesh->minimumExtent[2])};
    Vec3Df fMaxExtent = {static_cast<float>(diskMesh->maximumExtent[0]),
                         static_cast<float>(diskMesh->maximumExtent[1]),
                         static_cast<float>(diskMesh->maximumExtent[2])};
    float gridDelta = static_cast<float>(context.domain->getGridDelta());

    const viennaray::gpu::LineMesh lineMeshRay{
        fNodes,     surfaceMesh_->lines, fLineNormals,
        fMinExtent, fMaxExtent,          gridDelta};

    rayTracer_.setGeometry(lineMeshRay);

    auto model =
        std::dynamic_pointer_cast<gpu::ProcessModelGPU<NumericType, D>>(
            context.model);
    if (model->useMaterialIds()) {
      auto const &pointMaterialIds =
          diskMesh->getCellData().getScalarData("MaterialIds");
      std::vector<int> lineMaterialIds(surfaceMesh_->lines.size());
      auto &pointKdTree = context.translationField->getKdTree();
      if (pointKdTree->getNumberOfPoints() != diskMesh->nodes.size()) {
        pointKdTree->setPoints(diskMesh->nodes);
        pointKdTree->build();
      }
      // convert disk material ids to element material ids
      for (int i = 0; i < elementCenters.size(); i++) {
        auto closestPoint = pointKdTree->findNearest(elementCenters[i]);
        lineMaterialIds[i] =
            static_cast<int>(pointMaterialIds->at(closestPoint->first));
      }
      rayTracer_.setMaterialIds(lineMaterialIds);
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

    std::vector<Vec3D<NumericType>> elementCenters(surfaceMesh_->lines.size());
    for (int i = 0; i < surfaceMesh_->lines.size(); ++i) {
      Vec3D<NumericType> p0 = {
          surfaceMesh_->nodes[surfaceMesh_->lines[i][0]][0],
          surfaceMesh_->nodes[surfaceMesh_->lines[i][0]][1],
          surfaceMesh_->nodes[surfaceMesh_->lines[i][0]][2]};
      Vec3D<NumericType> p1 = {
          surfaceMesh_->nodes[surfaceMesh_->lines[i][1]][0],
          surfaceMesh_->nodes[surfaceMesh_->lines[i][1]][1],
          surfaceMesh_->nodes[surfaceMesh_->lines[i][1]][2]};
      elementCenters[i] = (p0 + p1) / NumericType(2);
    }

    auto diskMesh_ = *context.diskMesh;
    CudaBuffer d_coverages; // device buffer for coverages
    if (context.flags.useCoverages) {
      auto coverages = model->getSurfaceModel()->getCoverages();
      assert(coverages);
      assert(context.diskMesh);
      auto numCov = coverages->getScalarDataSize();
      auto &pointKdTree = context.translationField->getKdTree();
      if (pointKdTree->getNumberOfPoints() != diskMesh_.nodes.size()) {
        pointKdTree->setPoints(diskMesh_.nodes);
        pointKdTree->build();
      }

      // Convert disk coverages to element coverages
      // PointToElementData<NumericType, float>(d_coverages, coverages,
      //                                        pointKdTree, surfaceMesh_)
      //     .apply();
      std::vector<float> cov(surfaceMesh_->lines.size() * numCov, 0.f);

      for (int i = 0; i < numCov; ++i) {
        std::vector<NumericType> temp = *(coverages->getScalarData(i));
        std::vector<float> tempCasted(temp.begin(), temp.end());
        assert(tempCasted.size() == diskMesh_.getNodes().size());
        for (int j = 0; j < elementCenters.size(); j++) {
          auto closestPoint = pointKdTree->findNearest(elementCenters[j]);
          cov[j + i * surfaceMesh_->lines.size()] =
              tempCasted[closestPoint->first];
        }
      }
      d_coverages.allocUpload(cov);
      rayTracer_.setElementData(d_coverages, numCov);
    }

    // run the ray tracer
    rayTracer_.apply();
    downloadResultsToPointData(*fluxes, context.diskMesh,
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
                                 context.diskMesh,
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

  void
  downloadResultsToPointData(viennals::PointData<NumericType> &pointData,
                             SmartPointer<viennals::Mesh<NumericType>> diskMesh,
                             int smoothingNeighbors) {
    const auto numRates = rayTracer_.getNumberOfRates();
    const auto numPoints = rayTracer_.getNumberOfElements();
    auto diskMesh_ = *diskMesh;
    const auto numDisks = diskMesh_.nodes.size();
    assert(numRates > 0);
    auto particles = rayTracer_.getParticles();

    for (int pIdx = 0; pIdx < particles.size(); pIdx++) {
      for (int dIdx = 0; dIdx < particles[pIdx].dataLabels.size(); dIdx++) {
        std::vector<float> elementFlux(numPoints);
        rayTracer_.getFlux(elementFlux.data(), pIdx, dIdx, smoothingNeighbors);
        auto name = particles[pIdx].dataLabels[dIdx];

        // convert line fluxes to disk fluxes
        std::vector<float> diskFlux(numDisks, 0.f);

        for (int i = 0; i < numDisks; i++) {
          auto closestPoint = elementKdTree_->findNearest(diskMesh_.nodes[i]);
          diskFlux[i] = elementFlux[closestPoint->first];
        }
        // TODO: maybe smooth disk fluxes here additionally

        std::vector<NumericType> diskFluxCasted(diskFlux.begin(),
                                                diskFlux.end());
        pointData.insertReplaceScalarData(std::move(diskFluxCasted), name);
      }
    }
  }

private:
  std::shared_ptr<DeviceContext> deviceContext_;
  viennaray::gpu::TraceLine<NumericType, D> rayTracer_;

  KDTreeType elementKdTree_;
  MeshType surfaceMesh_;

  bool rayTracerInitialized_ = false;
};

} // namespace viennaps

#endif
