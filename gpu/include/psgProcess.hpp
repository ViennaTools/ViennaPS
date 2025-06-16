#pragma once

#include <vcContext.hpp>

#include <lsAdvect.hpp>
#include <lsDomain.hpp>
#include <lsMesh.hpp>

#include <psDomain.hpp>
#include <psProcessBase.hpp>

#include <raygTrace.hpp>

#include "psgCreateSurfaceMesh.hpp"
#include "psgElementToPointData.hpp"
#include "psgPointToElementData.hpp"
#include "psgProcessModel.hpp"

#include <cassert>

namespace viennaps::gpu {

using namespace viennacore;

template <typename NumericType, int D>
class Process final : public ProcessBase<NumericType, D> {
  using DomainType = SmartPointer<Domain<NumericType, D>>;
  using ModelType = SmartPointer<ProcessModel<NumericType, D>>;
  using TranslatorType = std::unordered_map<unsigned long, unsigned long>;
  using KDTreeType =
      SmartPointer<KDTree<NumericType, std::array<NumericType, 3>>>;
  using MeshType = SmartPointer<viennals::Mesh<float>>;

public:
  Process(Context &context) : context_(context), rayTracer_(context) {}
  Process(Context &context, DomainType domain)
      : ProcessBase<NumericType, D>(domain), context_(context),
        rayTracer_(context) {}
  Process(Context &context, DomainType domain, ModelType processModel,
          NumericType duration = 0.0)
      : ProcessBase<NumericType, D>(domain, processModel, duration),
        context_(context), rayTracer_(context), processModel_(processModel) {}

  // Set the process model. This can be either a pre-configured process model or
  // a custom process model. A custom process model must interface the
  // pscProcessModel class.
  void setProcessModel(ModelType processModel) {
    processModel_ = processModel;
    this->model_ = processModel;
    rayTracerInitialized_ = false;
  }

protected:
  bool checkInput() override {
    const auto name = processModel_->getProcessName().value_or("default");

    if (processModel_->getParticleTypes().empty()) {
      Logger::getInstance()
          .addWarning("No particle types in process model: " + name)
          .print();
      return false;
    }

    if (processModel_->getPipelineFileName().empty()) {
      Logger::getInstance()
          .addWarning("No pipeline in process model: " + name)
          .print();
      return false;
    }

    return true;
  }

  void initFluxEngine() override {
    if (!rayTracerInitialized_) {
      // Check for periodic boundary conditions
      bool periodicBoundary = false;
      if (rayTracingParams_.ignoreFluxBoundaries) {
        Logger::getInstance()
            .addWarning("Ignoring flux boundaries not implemented on GPU.")
            .print();
      } else {
        for (unsigned i = 0; i < D; ++i) {
          if (domain_->getGrid().getBoundaryConditions(i) ==
              viennals::BoundaryConditionEnum::PERIODIC_BOUNDARY) {
            periodicBoundary = true;
            break;
          }
        }
      }
      rayTracer_.setPipeline(processModel_->getPipelineFileName(),
                             context_.modulePath);
      rayTracer_.setNumberOfRaysPerPoint(rayTracingParams_.raysPerPoint);
      rayTracer_.setUseRandomSeeds(rayTracingParams_.useRandomSeeds);
      rayTracer_.setPeriodicBoundary(periodicBoundary);
      for (auto &particle : processModel_->getParticleTypes()) {
        rayTracer_.insertNextParticle(particle);
      }
      rayTracer_.prepareParticlePrograms();
    }
    rayTracer_.setParameters(processModel_->getProcessDataDPtr());
    rayTracerInitialized_ = true;
  }

  void setFluxEngineGeometry() override {
    surfaceMesh_ = viennals::Mesh<float>::New();
    if (!elementKdTree_)
      elementKdTree_ = KDTreeType::New();
    CreateSurfaceMesh<NumericType, float, D>(domain_->getLevelSets().back(),
                                             surfaceMesh_, elementKdTree_)
        .apply();

    auto mesh = CreateTriangleMesh(domain_->getGridDelta(), surfaceMesh_);
    rayTracer_.setGeometry(mesh);

    if (processModel_->useMaterialIds()) {
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
  }

  SmartPointer<viennals::PointData<NumericType>>
  calculateFluxes(const bool useCoverages,
                  const bool useProcessParams) override {

    const auto name = processModel_->getProcessName().value_or("default");
    const auto logLevel = Logger::getLogLevel();

    CudaBuffer d_coverages; // device buffer for coverages
    if (useCoverages) {
      auto coverages = processModel_->getSurfaceModel()->getCoverages();
      assert(coverages);
      assert(diskMesh_);
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
    auto fluxes = viennals::PointData<NumericType>::New();
    ElementToPointData<NumericType, float>(
        rayTracer_.getResults(), fluxes, rayTracer_.getParticles(),
        elementKdTree_, diskMesh_, surfaceMesh_,
        domain_->getGridDelta() * rayTracingParams_.smoothingNeighbors)
        .apply();

    // output
    if (logLevel >= static_cast<unsigned>(LogLevel::INTERMEDIATE)) {
      if (useCoverages) {
        auto coverages = processModel_->getSurfaceModel()->getCoverages();
        downloadCoverages(d_coverages, surfaceMesh_->getCellData(), coverages,
                          surfaceMesh_->getElements<3>().size());
      }
      downloadResultsToPointData(surfaceMesh_->getCellData());
      static unsigned iterations = 0;
      viennals::VTKWriter<float>(
          surfaceMesh_, name + "_flux_" + std::to_string(iterations++) + ".vtp")
          .apply();
    }

    if (useCoverages) {
      d_coverages.free();
    }

    return fluxes;
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
  Context &context_;
  viennaray::gpu::Trace<NumericType, D> rayTracer_;

  using ProcessBase<NumericType, D>::domain_;
  using ProcessBase<NumericType, D>::rayTracingParams_;
  using ProcessBase<NumericType, D>::diskMesh_;
  using ProcessBase<NumericType, D>::translationField_;
  ModelType processModel_;
  KDTreeType elementKdTree_;
  MeshType surfaceMesh_;

  bool coveragesInitialized_ = false;
  bool rayTracerInitialized_ = false;
};

} // namespace viennaps::gpu
