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

template <typename CellType, typename NumericType, int D>
class psProcess
{
  using translatorType = std::unordered_map<unsigned long, unsigned long>;
  psSmartPointer<psDomain<CellType, NumericType, D>> domain = nullptr;
  psSmartPointer<psProcessModel<NumericType>> model = nullptr;
  double processDuration;
  rayTraceDirection sourceDirection;
  int raysPerPoint = 1000;

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

public:
  void apply()
  {
    double remainingTime = processDuration;
    const NumericType gridDelta =
        domain->getLevelSets.back().getGrid().getGridDelta();

    auto diskMesh = lsSmartPointer<lsMesh<NumericType>>::New();
    auto translator = lsSmartPointer<translatorType>::New();
    lsToDiskMesh<NumericType, D> meshConverter(domain->getLevelSets().back(),
                                               diskMesh, translator);
    lsAdvect<NumericType, D> advectionKernel;
    for (auto &dom : domain->getLevelSets())
    {
      meshConverter.inserNextLevelSet(dom);
      advectionKernel.inserNextLevelSet(dom);
    }
    advectionKernel->setVelocityField(model->getVelocityField());

    /* --------- Setup for ray tracing ----------- */
    // Map the domain boundary to the ray tracing boundaries
    rayTraceBoundary rayBoundCond[D];
    for (unsigned i = 0; i < D; ++i)
      rayBoundCond[i] = convertBoundaryCondition(
          domain.getLevelSets.back()->getGrid()->getBoundaryCondition(i));

    rayTrace<NumericType, D> rayTrace;
    rayTrace.setSourceDirection(sourceDirection);
    rayTrace.setNumberOfRaysPerPoint(raysPerPoint);
    rayTrace.setBoundaryConditions(rayBoundCond);
    rayTrace.setCalculateFlux(false);

    // Initialize coverages
    {
      meshConverter.apply();
      auto numPoints = diskMesh->getNodes().size();
      model->getSurfaceModel()->initializeCoverages(numPoints);
      auto points = diskMesh->getNodes();
      auto normals = *diskMesh->getCellData("Normals");
      auto materialIds = *diskMesh->getCellData("MaterialIds");
      rayTrace.setGeometry(points, normals, gridDelta);
      rayTrace.setMaterialIDs(materialIds);

      rayTracingData<NumericType> rayTraceCoverages = movePointDataToRayData(model->getSurfaceModel()->getCoverages());
      rayTrace.setGlobalData(rayTraceCoverages);
      // all coverages are moved into rayTracingData and are thus invalidated in the surfaceModel
      model->getSurfaceModel()->getCoverages()->clear();

      auto Rates = psSmartPointer<psPointData<NumericType>>::New();

      for (auto &particle : model->getParticleTypes())
      {
        rayTrace.setParticleType(particle);
        rayTrace.apply();

        // fill up rates vector with rates from this particle type
        auto numRates = particle.getRequiredLocalDataSize();
        auto &localData = rayTrace.getLocalData();
        for (int i = 0; i < numRates; ++i)
        {
          Rates->insertNextScalarData(std::move(localData.getVectorData(i)), localData.getVectorDataLabel(i));
        }
      }

      model->getSurfaceModel()->updateCoverages(Rates);
    }

    while (remainingTime > 0.)
    {
      std::vector<std::vector<NumericType>> Rates;

      meshConverter.apply();
      auto points = diskMesh->getNodes();
      auto normals = *diskMesh->getCellData("Normals");
      auto materialIds = *diskMesh->getCellData("MaterialIds");
      rayTrace.setGeometry(points, normals, gridDelta);
      rayTrace.setMaterialIDs(materialIds);
      // rayTrace.setGlobalData(
      //     model->getSurfaceModel()
      //         ->getCoverages()); // just set as a reference (using shared ptr)

      for (auto &particle : model->getParticleTypes())
      {
        rayTrace.setParticleType(particle);
        rayTrace.apply();

        // fill up rates vector with rates from this particle type
        for (auto &rate : rayTrace.getLocalData().getVectorData())
        {
          Rates.push_back(std::move(rate));
        }
      }

      auto velField = model->getVelocityField();
      velField->setVelocities(
          model->getSurfaceModel()->calculateVelocities(Rates, materialIds));

      advectionKernel->setAdvectionTime(remainingTime);
      advectionKernel->apply();
      remainingTime -= advectionKernel->getAdvectedTime();
    }
  }

  template <typename ProcessModelType>
  void setProcessModel(psSmartPointer<ProcessModelType> passedProcessModel)
  {
    model = std::dynamic_pointer_cast<psProcessModel<NumericType>>(passedProcessModel);
  }
};

#endif