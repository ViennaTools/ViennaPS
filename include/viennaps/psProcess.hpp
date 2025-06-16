#pragma once

#include "psProcessBase.hpp"
#include "psProcessModel.hpp"
#include "psTranslationField.hpp"
#include "psUtil.hpp"

#include <lsAdvect.hpp>

#include <rayTrace.hpp>

namespace viennaps {

using namespace viennacore;

/// This class server as the main process tool, applying a user- or pre-defined
/// process model to a domain. Depending on the user inputs surface advection, a
/// single callback function or a geometric advection is applied.
template <typename NumericType, int D>
class Process final : public ProcessBase<NumericType, D> {
  // Typedefs
  using TranslatorType = std::unordered_map<unsigned long, unsigned long>;
  using DomainType = SmartPointer<Domain<NumericType, D>>;

public:
  Process() = default;
  Process(DomainType domain) : ProcessBase<NumericType, D>(domain) {}
  // Constructor for a process with a pre-configured process model.
  Process(DomainType domain,
          SmartPointer<ProcessModel<NumericType, D>> processModel,
          const NumericType duration = 0.)
      : ProcessBase<NumericType, D>(domain, processModel, duration),
        processModel_(processModel) {}

  // Set the process model. This can be either a pre-configured process model or
  // a custom process model. A custom process model must interface the
  // psProcessModel class.
  void
  setProcessModel(SmartPointer<ProcessModel<NumericType, D>> processModel) {
    processModel_ = processModel;
    this->model_ = processModel;
  }

  void setRayTracingDiskRadius(NumericType radius) {
    rayTracingParams_.diskRadius = radius;
    if (rayTracingParams_.diskRadius < 0.) {
      Logger::getInstance()
          .addWarning("Disk radius must be positive. Using default value.")
          .print();
      rayTracingParams_.diskRadius = 0.;
    }
  }

  void writeParticleDataLogs(const std::string &fileName) {
    std::ofstream file(fileName.c_str());

    for (std::size_t i = 0; i < particleDataLogs.size(); i++) {
      if (!particleDataLogs[i].data.empty()) {
        file << "particle" << i << "_data ";
        for (std::size_t j = 0; j < particleDataLogs[i].data[0].size(); j++) {
          file << particleDataLogs[i].data[0][j] << " ";
        }
        file << "\n";
      }
    }

    file.close();
  }

protected:
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

    return std::move(rayData);
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

  bool checkInput() override { return true; }

  void initFluxEngine() override {
    // Map the domain boundary to the ray tracing boundaries
    viennaray::BoundaryCondition rayBoundaryCondition[D];
    if (rayTracingParams_.ignoreFluxBoundaries) {
      for (unsigned i = 0; i < D; ++i)
        rayBoundaryCondition[i] = viennaray::BoundaryCondition::IGNORE;
    } else {
      for (unsigned i = 0; i < D; ++i)
        rayBoundaryCondition[i] = util::convertBoundaryCondition(
            domain_->getGrid().getBoundaryConditions(i));
    }
    rayTracer.setBoundaryConditions(rayBoundaryCondition);
    rayTracer.setSourceDirection(rayTracingParams_.sourceDirection);
    rayTracer.setNumberOfRaysPerPoint(rayTracingParams_.raysPerPoint);
    rayTracer.setUseRandomSeeds(rayTracingParams_.useRandomSeeds);
    rayTracer.setCalculateFlux(false);

    if (auto source = processModel_->getSource()) {
      rayTracer.setSource(source);
      Logger::getInstance().addInfo("Using custom source.").print();
    }
    if (auto primaryDirection = processModel_->getPrimaryDirection()) {
      Logger::getInstance()
          .addInfo("Using primary direction: " +
                   util::arrayToString(primaryDirection.value()))
          .print();
      rayTracer.setPrimaryDirection(primaryDirection.value());
    }

    // initialize particle data logs
    particleDataLogs.resize(processModel_->getParticleTypes().size());
    for (std::size_t i = 0; i < processModel_->getParticleTypes().size(); i++) {
      if (int logSize = processModel_->getParticleLogSize(i); logSize > 0) {
        particleDataLogs[i].data.resize(1);
        particleDataLogs[i].data[0].resize(logSize);
      }
    }
  }

  void setFluxEngineGeometry() override {
    assert(diskMesh_ != nullptr);
    auto points = diskMesh_->getNodes();
    auto normals = *diskMesh_->getCellData().getVectorData("Normals");
    auto materialIds = *diskMesh_->getCellData().getScalarData("MaterialIds");

    if (rayTracingParams_.diskRadius == 0.) {
      rayTracer.setGeometry(points, normals, domain_->getGrid().getGridDelta());
    } else {
      rayTracer.setGeometry(points, normals, domain_->getGrid().getGridDelta(),
                            rayTracingParams_.diskRadius);
    }
    rayTracer.setMaterialIds(materialIds);
  }

  SmartPointer<viennals::PointData<NumericType>>
  calculateFluxes(const bool useCoverages,
                  const bool useProcessParams) override {

    viennaray::TracingData<NumericType> rayTracingData;
    auto surfaceModel = this->model_->getSurfaceModel();

    // move coverages to the ray tracer
    if (useCoverages) {
      rayTracingData = movePointDataToRayData(surfaceModel->getCoverages());
    }

    if (useProcessParams) {
      // store scalars in addition to coverages
      auto processParams = surfaceModel->getProcessParameters();
      NumericType numParams = processParams->getScalarData().size();
      rayTracingData.setNumberOfScalarData(numParams);
      for (size_t i = 0; i < numParams; ++i) {
        rayTracingData.setScalarData(i, processParams->getScalarData(i),
                                     processParams->getScalarDataLabel(i));
      }
    }

    if (useCoverages || useProcessParams)
      rayTracer.setGlobalData(rayTracingData);

    auto fluxes = runRayTracer();

    // move coverages back in the model
    if (useCoverages)
      moveRayDataToPointData(surfaceModel->getCoverages(), rayTracingData);

    return fluxes;
  }

private:
  SmartPointer<viennals::PointData<NumericType>> runRayTracer() {
    auto fluxes = viennals::PointData<NumericType>::New();
    unsigned particleIdx = 0;
    for (auto &particle : processModel_->getParticleTypes()) {
      int dataLogSize = processModel_->getParticleLogSize(particleIdx);
      if (dataLogSize > 0) {
        rayTracer.getDataLog().data.resize(1);
        rayTracer.getDataLog().data[0].resize(dataLogSize, 0.);
      }
      rayTracer.setParticleType(particle);
      rayTracer.apply();

      // fill up fluxes vector with fluxes from this particle type
      auto &localData = rayTracer.getLocalData();
      int numFluxes = particle->getLocalDataLabels().size();
      for (int i = 0; i < numFluxes; ++i) {
        auto flux = std::move(localData.getVectorData(i));

        // normalize and smooth
        rayTracer.normalizeFlux(flux, rayTracingParams_.normalizationType);
        if (rayTracingParams_.smoothingNeighbors > 0)
          rayTracer.smoothFlux(flux, rayTracingParams_.smoothingNeighbors);

        fluxes->insertNextScalarData(std::move(flux),
                                     localData.getVectorDataLabel(i));
      }

      if (dataLogSize > 0) {
        particleDataLogs[particleIdx].merge(rayTracer.getDataLog());
      }
      ++particleIdx;
    }
    return fluxes;
  }

private:
  // Members
  using ProcessBase<NumericType, D>::domain_;
  using ProcessBase<NumericType, D>::rayTracingParams_;
  using ProcessBase<NumericType, D>::diskMesh_;

  SmartPointer<ProcessModel<NumericType, D>> processModel_;
  viennaray::Trace<NumericType, D> rayTracer;
  std::vector<viennaray::DataLog<NumericType>> particleDataLogs;
};

} // namespace viennaps