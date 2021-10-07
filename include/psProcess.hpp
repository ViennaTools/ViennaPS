#ifndef PS_PROCESS
#define PS_PROCESS

#include <lsAdvect.hpp>
#include <lsDomain.hpp>
#include <lsMesh.hpp>
#include <lsToDiskMesh.hpp>
#include <psDomain.hpp>
#include <psProcessModel.hpp>
#include <psSmartPointer.hpp>
#include <psSurfaceModel.hpp>
#include <psTranslationField.hpp>
#include <psVelocityField.hpp>
#include <psVolumeModel.hpp>
#include <rayBoundCondition.hpp>
#include <rayParticle.hpp>
#include <rayTrace.hpp>

template <typename CellType, typename NumericType, int D> class psProcess {
  using translatorType = std::unordered_map<unsigned long, unsigned long>;
  psSmartPointer<psDomain<CellType, NumericType, D>> domain = nullptr;
  psSmartPointer<psProcessModel<NumericType>> model = nullptr;
  double processDuration;
  rayTraceDirection sourceDirection;
  long raysPerPoint = 2000;

  void printLS(lsSmartPointer<lsDomain<NumericType, D>> dom, std::string name) {
    auto mesh = lsSmartPointer<lsMesh<NumericType>>::New();
    lsToSurfaceMesh<NumericType, D>(dom, mesh).apply();
    lsVTKWriter<NumericType>(mesh, name).apply();
  }

  rayTraceBoundary convertBoundaryCondition(
      lsBoundaryConditionEnum<D> originalBoundaryCondition) {
    switch (originalBoundaryCondition) {
    case lsBoundaryConditionEnum<D>::REFLECTIVE_BOUNDARY:
      return rayTraceBoundary::REFLECTIVE;

    case lsBoundaryConditionEnum<D>::INFINITE_BOUNDARY:
      return rayTraceBoundary::IGNORE;

    case lsBoundaryConditionEnum<D>::PERIODIC_BOUNDARY:
      return rayTraceBoundary::PERIODIC;

    case lsBoundaryConditionEnum<D>::POS_INFINITE_BOUNDARY:
      return rayTraceBoundary::IGNORE;

    case lsBoundaryConditionEnum<D>::NEG_INFINITE_BOUNDARY:
      return rayTraceBoundary::IGNORE;
    }
  }

  rayTracingData<NumericType>
  movePointDataToRayData(psSmartPointer<psPointData<NumericType>> pointData) {
    rayTracingData<NumericType> rayData;
    const auto numData = pointData->getScalarDataSize();
    rayData.setNumberOfVectorData(numData);
    for (size_t i = 0; i < numData; ++i) {
      auto label = pointData->getScalarDataLabel(i);
      rayData.setVectorData(i, std::move(*pointData->getScalarData(label)),
                            label);
    }

    return std::move(rayData);
  }

  void
  moveRayDataToPointData(psSmartPointer<psPointData<NumericType>> pointData,
                         rayTracingData<NumericType> &rayData) {
    pointData->clear();
    const auto numData = rayData.getVectorData().size();
    for (size_t i = 0; i < numData; ++i)
      pointData->insertNextScalarData(std::move(rayData.getVectorData(i)),
                                      rayData.getVectorDataLabel(i));
  }

public:
  void apply() {
    /* ---------- Process Setup --------- */
    auto name = model->getProcessName();
    double remainingTime = processDuration;
    const NumericType gridDelta =
        domain->getLevelSets()->back()->getGrid().getGridDelta();

    auto diskMesh = lsSmartPointer<lsMesh<NumericType>>::New();
    auto translator = lsSmartPointer<translatorType>::New();
    lsToDiskMesh<NumericType, D> meshConverter(diskMesh);
    meshConverter.setTranslator(translator);

    auto transField = psSmartPointer<psTranslationField<NumericType>>::New();
    transField->setTranslator(translator);
    transField->setVelocityField(model->getVelocityField());

    lsAdvect<NumericType, D> advectionKernel;
    advectionKernel.setVelocityField(transField);

    for (auto dom : *domain->getLevelSets()) {
      meshConverter.insertNextLevelSet(dom);
      advectionKernel.insertNextLevelSet(dom);
    }

    /* --------- Setup for ray tracing ----------- */
    bool useRayTracing = true;
    if (model->getParticleTypes() == nullptr)
      useRayTracing = false;

    rayTraceBoundary rayBoundCond[D];
    rayTrace<NumericType, D> rayTrace;

    if (useRayTracing) {
      // Map the domain boundary to the ray tracing boundaries
      for (unsigned i = 0; i < D; ++i)
        rayBoundCond[i] = convertBoundaryCondition(
            domain->getGrid().getBoundaryConditions(i));

      rayTrace.setSourceDirection(sourceDirection);
      rayTrace.setNumberOfRaysPerPoint(raysPerPoint);
      rayTrace.setBoundaryConditions(rayBoundCond);
      rayTrace.setCalculateFlux(false);
    }
    bool useCoverages = false;
    // Initialize coverages
    {
      meshConverter.apply();
      auto numPoints = diskMesh->getNodes().size();
      model->getSurfaceModel()->initializeCoverages(numPoints);
      if (model->getSurfaceModel()->getCoverages() != nullptr) {
        useCoverages = true;
        auto points = diskMesh->getNodes();
        auto normals = *diskMesh->getCellData().getVectorData("Normals");
        auto materialIds =
            *diskMesh->getCellData().getScalarData("MaterialIds");
        rayTrace.setGeometry(points, normals, gridDelta);
        rayTrace.setMaterialIds(materialIds);

        rayTracingData<NumericType> rayTraceCoverages =
            movePointDataToRayData(model->getSurfaceModel()->getCoverages());
        rayTrace.setGlobalData(rayTraceCoverages);

        auto Rates = psSmartPointer<psPointData<NumericType>>::New();

        for (auto &particle : *model->getParticleTypes()) {
          rayTrace.setParticleType(particle);
          rayTrace.apply();

          // fill up rates vector with rates from this particle type
          auto discAreas = rayTrace.getDiscAreas();
          auto numRates = particle->getRequiredLocalDataSize();
          auto &localData = rayTrace.getLocalData();
          for (int i = 0; i < numRates; ++i) {
            auto rate = std::move(localData.getVectorData(i));

            // TODO: normalize rates
            // NumericType sum = 0;
            // for (size_t j = 0; j < rate.size(); j++)
            // {
            //   rate[i] /= discAreas[i];
            //   sum += rate[i];
            // }
            // for (size_t j = 0; j < rate.size(); j++)
            // {
            //   rate[i] /= sum;
            // }

            Rates->insertNextScalarData(std::move(rate),
                                        localData.getVectorDataLabel(i));
          }
        }

        // move coverages back in the model
        moveRayDataToPointData(model->getSurfaceModel()->getCoverages(),
                               rayTraceCoverages);
        model->getSurfaceModel()->updateCoverages(
            Rates, raysPerPoint * materialIds.size());
      }
    }

    size_t counter = 0;
    while (remainingTime > 0.) {
      std::cout << name << " remaining time: " << remainingTime << std::endl;
      auto Rates = psSmartPointer<psPointData<NumericType>>::New();

      meshConverter.apply();
      auto materialIds = *diskMesh->getCellData().getScalarData("MaterialIds");
      if (useRayTracing) {
        auto points = diskMesh->getNodes();
        auto normals = *diskMesh->getCellData().getVectorData("Normals");
        rayTrace.setGeometry(points, normals, gridDelta);
        rayTrace.setMaterialIds(materialIds);

        rayTracingData<NumericType> rayTraceCoverages;
        if (useCoverages) {
          rayTraceCoverages =
              movePointDataToRayData(model->getSurfaceModel()->getCoverages());
          rayTrace.setGlobalData(rayTraceCoverages);
        }

        for (auto &particle : *model->getParticleTypes()) {
          rayTrace.setParticleType(particle);
          rayTrace.apply();

          // fill up rates vector with rates from this particle type
          auto discAreas = rayTrace.getDiscAreas();
          auto numRates = particle->getRequiredLocalDataSize();
          auto &localData = rayTrace.getLocalData();
          for (int i = 0; i < numRates; ++i) {
            auto rate = std::move(localData.getVectorData(i));

            // TODO: normalize rates
            // NumericType sum = 0;
            // for (size_t j = 0; j < rate.size(); j++)
            // {
            //   rate[i] /= discAreas[i];
            //   sum += rate[i];
            // }
            // for (size_t j = 0; j < rate.size(); j++)
            // {
            //   rate[i] /= sum;
            // }

            Rates->insertNextScalarData(std::move(rate),
                                        localData.getVectorDataLabel(i));
          }
        }

        if (useCoverages)
          moveRayDataToPointData(model->getSurfaceModel()->getCoverages(),
                                 rayTraceCoverages);
      }

      auto velocitites = model->getSurfaceModel()->calculateVelocities(
          Rates, materialIds, raysPerPoint * materialIds.size());
      model->getVelocityField()->setVelocities(velocitites);
      advectionKernel.apply();
      remainingTime -= advectionKernel.getAdvectedTime();
    }
  }

  template <typename ProcessModelType>
  void setProcessModel(psSmartPointer<ProcessModelType> passedProcessModel) {
    model = std::dynamic_pointer_cast<psProcessModel<NumericType>>(
        passedProcessModel);
  }

  void
  setDomain(psSmartPointer<psDomain<CellType, NumericType, D>> passedDomain) {
    domain = passedDomain;
  }

  /// Set the source direction, where the rays should be traced from.
  void setSourceDirection(const rayTraceDirection passedDirection) {
    sourceDirection = passedDirection;
  }

  void setProcessDuration(double passedDuration) {
    processDuration = passedDuration;
  }

  void setNumberOfRaysPerPoint(long numRays) { raysPerPoint = numRays; }
};

#endif