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
#include <psVelocityField.hpp>
#include <psVolumeModel.hpp>
#include <rayBoundCondition.hpp>
#include <rayParticle.hpp>
#include <rayTrace.hpp>
#include <lsDomain.hpp>

template <typename CellType, typename NumericType, int D>
class psProcess
{
  using translatorType = std::unordered_map<unsigned long, unsigned long>;
  psSmartPointer<psDomain<CellType, NumericType, D>> domain = nullptr;
  psSmartPointer<psProcessModel<NumericType>> model = nullptr;
  double processDuration;
  rayTraceDirection sourceDirection;
  long raysPerPoint = 2000;

  rayTraceBoundary convertBoundaryCondition(
      lsBoundaryConditionEnum<D> originalBoundaryCondition)
  {
    switch (originalBoundaryCondition)
    {
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

  rayTracingData<NumericType> movePointDataToRayData(psSmartPointer<psPointData<NumericType>> pointData)
  {
    rayTracingData<NumericType> rayData;
    const auto numCoverages = pointData->getScalarDataSize();
    rayData.setNumberOfVectorData(numCoverages);
    for (size_t i = 0; i < numCoverages; ++i)
    {
      auto label = pointData->getScalarDataLabel(i);
      rayData.setVectorData(i, std::move(*pointData->getScalarData(label)), label);
    }
    return std::move(rayData);
  }

  rayTracingData<NumericType> copyPointDataToRayData(psSmartPointer<psPointData<NumericType>> pointData)
  {
    rayTracingData<NumericType> rayData;
    const auto numCoverages = pointData->getScalarDataSize();
    rayData.setNumberOfVectorData(numCoverages);
    for (size_t i = 0; i < numCoverages; ++i)
    {
      auto label = pointData->getScalarDataLabel(i);
      rayData.setVectorData(i, *pointData->getScalarData(label), label);
    }
    return std::move(rayData);
  }

  class translationField : public lsVelocityField<NumericType>
  {
  public:
    translationField() = default;

    NumericType getScalarVelocity(const std::array<NumericType, 3> &coordinate,
                                  int material,
                                  const std::array<NumericType, 3> &normalVector,
                                  unsigned long pointId)
    {
      auto surfacePointId = translateLsId(pointId);
      if (surfacePointId != -1)
      {
        return modelVelocityField->getScalarVelocity(coordinate, material, normalVector, surfacePointId);
      }
      else
      {
        // TODO: probably should throw a warning that something went wrong here
        return 0;
      }
    }

    std::array<NumericType, 3>
    getVectorVelocity(const std::array<NumericType, 3> &coordinate, int material,
                      const std::array<NumericType, 3> &normalVector,
                      unsigned long pointId)
    {
      auto surfacePointId = translateLsId(pointId);
      if (surfacePointId != -1)
      {
        return modelVelocityField->getVectorVelocity(coordinate, material, normalVector, surfacePointId);
      }
      else
      {
        // TODO: probably should throw a warning that something went wrong here
        return {0., 0., 0.};
      }
    }

    NumericType getDissipationAlpha(int direction, int material,
                                    const std::array<NumericType, 3> &centralDifferences)
    {
      return modelVelocityField->getDissipationAlpha(direction, material, centralDifferences);
    }

    void setTranslator(psSmartPointer<translatorType> passedTranslator)
    {
      translator = passedTranslator;
    }

    void setVelocityField(psSmartPointer<psVelocityField<NumericType>> passedVeloField)
    {
      modelVelocityField = passedVeloField;
    }

  private:
    long translateLsId(unsigned long lsId)
    {
      if (auto it = translator->find(lsId); it != translator->end())
      {
        return it->second;
      }
      else
      {
        return -1;
      }
    }

    psSmartPointer<translatorType> translator;
    psSmartPointer<psVelocityField<NumericType>> modelVelocityField;
  };

public:
  void apply()
  {
    double remainingTime = processDuration;
    const NumericType gridDelta =
        domain->getLevelSets()->back()->getGrid().getGridDelta();

    auto diskMesh = lsSmartPointer<lsMesh<NumericType>>::New();
    auto translator = lsSmartPointer<translatorType>::New();
    lsToDiskMesh<NumericType, D> meshConverter(diskMesh);
    meshConverter.setTranslator(translator);

    auto transField = psSmartPointer<translationField>::New();
    transField->setVelocityField(model->getVelocityField());

    lsAdvect<NumericType, D> advectionKernel;
    advectionKernel.setVelocityField(transField);

    for (auto dom : *domain->getLevelSets())
    {
      meshConverter.insertNextLevelSet(dom);
      advectionKernel.insertNextLevelSet(dom);
    }

    /* --------- Setup for ray tracing ----------- */
    // Map the domain boundary to the ray tracing boundaries
    rayTraceBoundary rayBoundCond[D];
    for (unsigned i = 0; i < D; ++i)
      rayBoundCond[i] = convertBoundaryCondition(
          domain->getGrid().getBoundaryConditions(i));

    rayTrace<NumericType, D> rayTrace;
    rayTrace.setSourceDirection(sourceDirection);
    rayTrace.setNumberOfRaysPerPoint(raysPerPoint);
    rayTrace.setBoundaryConditions(rayBoundCond);
    rayTrace.setCalculateFlux(false);

    bool useCoverages = false;
    // Initialize coverages
    {
      meshConverter.apply();
      auto numPoints = diskMesh->getNodes().size();
      model->getSurfaceModel()->initializeCoverages(numPoints);
      if (model->getSurfaceModel()->getCoverages() != nullptr)
      {
        useCoverages = true;
        auto points = diskMesh->getNodes();
        auto normals = *diskMesh->getCellData().getVectorData("Normals");
        auto materialIds = *diskMesh->getCellData().getScalarData("MaterialIds");
        rayTrace.setGeometry(points, normals, gridDelta);
        rayTrace.setMaterialIds(materialIds);

        rayTracingData<NumericType> rayTraceCoverages = movePointDataToRayData(model->getSurfaceModel()->getCoverages());
        rayTrace.setGlobalData(rayTraceCoverages);
        // all coverages are moved into rayTracingData and are thus invalidated in the surfaceModel
        model->getSurfaceModel()->getCoverages()->clear();

        auto Rates = psSmartPointer<psPointData<NumericType>>::New();

        for (auto &particle : *model->getParticleTypes())
        {
          rayTrace.setParticleType(particle);
          rayTrace.apply();

          // fill up rates vector with rates from this particle type
          auto numRates = particle->getRequiredLocalDataSize();
          auto &localData = rayTrace.getLocalData();
          for (int i = 0; i < numRates; ++i)
          {
            Rates->insertNextScalarData(std::move(localData.getVectorData(i)), localData.getVectorDataLabel(i));
          }
        }

        // TODO move ray data back into model

        model->getSurfaceModel()->updateCoverages(Rates, raysPerPoint);
      }
    }

    unsigned counter = 0;
    while (remainingTime > 0.)
    {
      std::cout << "Remaining time " << remainingTime << std::endl;
      auto Rates = psSmartPointer<psPointData<NumericType>>::New();

      meshConverter.apply();
      auto points = diskMesh->getNodes();
      auto normals = *diskMesh->getCellData().getVectorData("Normals");
      auto materialIds = *diskMesh->getCellData().getScalarData("MaterialIds");
      rayTrace.setGeometry(points, normals, gridDelta);
      rayTrace.setMaterialIds(materialIds);
      if (useCoverages)
      {
        // TODO: change this to move
        rayTracingData<NumericType> rayTraceCoverages = copyPointDataToRayData(model->getSurfaceModel()->getCoverages());
        rayTrace.setGlobalData(rayTraceCoverages);
      }

      for (auto &particle : *model->getParticleTypes())
      {
        rayTrace.setParticleType(particle);
        rayTrace.apply();

        // fill up rates vector with rates from this particle type
        auto numRates = particle->getRequiredLocalDataSize();
        auto &localData = rayTrace.getLocalData();
        for (int i = 0; i < numRates; ++i)
        {
          Rates->insertNextScalarData(std::move(localData.getVectorData(i)), localData.getVectorDataLabel(i));
        }
      }

      // move coverages back if necessary

      auto velocitites = model->getSurfaceModel()->calculateVelocities(Rates, materialIds, raysPerPoint);
      model->getVelocityField()->setVelocities(velocitites);
      diskMesh->getCellData().insertNextScalarData(*velocitites, "etchRate");
      lsVTKWriter<NumericType>(diskMesh, "VCEtch_disk_" + std::to_string(counter) + ".vtp").apply();

      // advectionKernel.setAdvectionTime(remainingTime);
      advectionKernel.apply();
      remainingTime -= advectionKernel.getAdvectedTime();
      ++counter;
    }
  }

  template <typename ProcessModelType>
  void setProcessModel(psSmartPointer<ProcessModelType> passedProcessModel)
  {
    model = std::dynamic_pointer_cast<psProcessModel<NumericType>>(passedProcessModel);
  }

  void setDomain(psSmartPointer<psDomain<CellType, NumericType, D>> passedDomain)
  {
    domain = passedDomain;
  }

  /// Set the source direction, where the rays should be traced from.
  void setSourceDirection(const rayTraceDirection passedDirection)
  {
    sourceDirection = passedDirection;
  }

  void setProcessDuration(double passedDuration)
  {
    processDuration = passedDuration;
  }
};

#endif