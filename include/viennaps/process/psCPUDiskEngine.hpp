#pragma once

#include "psFluxEngine.hpp"
#include "psProcessModel.hpp"

namespace viennaps {

using namespace viennacore;

/// This class server as the main process tool, applying a user- or pre-defined
/// process model to a domain. Depending on the user inputs surface advection, a
/// single callback function or a geometric advection is applied.
template <typename NumericType, int D>
class CPUDiskEngine final : public FluxEngine<NumericType, D> {
  // Typedefs
  using TranslatorType = std::unordered_map<unsigned long, unsigned long>;
  using DomainType = SmartPointer<Domain<NumericType, D>>;

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
    rayTracer_.setBoundaryConditions(rayBoundaryCondition);
    rayTracer_.setSourceDirection(context.rayTracingParams.sourceDirection);
    rayTracer_.setNumberOfRaysPerPoint(context.rayTracingParams.raysPerPoint);
    rayTracer_.setUseRandomSeeds(context.rayTracingParams.useRandomSeeds);
    rayTracer_.setCalculateFlux(false);

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

    // initialize particle data logs
    particleDataLogs_.resize(model->getParticleTypes().size());
    for (std::size_t i = 0; i < model->getParticleTypes().size(); i++) {
      if (int logSize = model->getParticleLogSize(i); logSize > 0) {
        particleDataLogs_[i].data.resize(1);
        particleDataLogs_[i].data[0].resize(logSize);
      }
    }

    return ProcessResult::SUCCESS;
  }

  ProcessResult updateSurface(ProcessContext<NumericType, D> &context) final {
    this->timer_.start();
    auto &diskMesh = context.diskMesh;
    assert(diskMesh != nullptr);

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

  ProcessResult
  calculateFluxes(ProcessContext<NumericType, D> &context,
                  viennacore::SmartPointer<viennals::PointData<NumericType>>
                      &fluxes) final {
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
    fluxes->clear();

    auto model = std::dynamic_pointer_cast<ProcessModelCPU<NumericType, D>>(
        context.model);

    unsigned particleIdx = 0;
    for (auto &particle : model->getParticleTypes()) {
      int dataLogSize = model->getParticleLogSize(particleIdx);
      if (dataLogSize > 0) {
        rayTracer_.getDataLog().data.resize(1);
        rayTracer_.getDataLog().data[0].resize(dataLogSize, 0.);
      }
      rayTracer_.setParticleType(particle);
      rayTracer_.apply();

      // fill up fluxes vector with fluxes from this particle type
      auto &localData = rayTracer_.getLocalData();
      int numFluxes = particle->getLocalDataLabels().size();
      for (int i = 0; i < numFluxes; ++i) {
        auto flux = std::move(localData.getVectorData(i));

        // normalize and smooth
        rayTracer_.normalizeFlux(flux,
                                 context.rayTracingParams.normalizationType);
        if (context.rayTracingParams.smoothingNeighbors > 0)
          rayTracer_.smoothFlux(flux,
                                context.rayTracingParams.smoothingNeighbors);

        fluxes->insertNextScalarData(std::move(flux),
                                     localData.getVectorDataLabel(i));
      }

      if (dataLogSize > 0) {
        particleDataLogs_[particleIdx].merge(rayTracer_.getDataLog());
      }
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

private:
  viennaray::Trace<NumericType, D> rayTracer_;
  std::vector<viennaray::DataLog<NumericType>> particleDataLogs_;

  /*
void writeParticleDataLogs(const std::string &fileName) {
std::ofstream file(fileName.c_str());

for (std::size_t i = 0; i < particleDataLogs.size(); i++) {
  if (!particleDataLogs[i].data.empty()) {
    file << "particle" << i << "_data\n";
    for (std::size_t j = 0; j < particleDataLogs[i].data[0].size(); j++) {
      file << particleDataLogs[i].data[0][j] << " ";
    }
    file << "\n";
  }
}

file.close();
}
SmartPointer<viennals::PointData<NumericType>> calculateFluxes() override {


}

private:

*/
};

} // namespace viennaps