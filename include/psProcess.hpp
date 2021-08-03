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
// #include <rayBoundCondition.hpp>
// #include <rayParticle.hpp>
#include <rayTrace.hpp>

template <typename CellType, typename NumericType, int D> class psProcess {
//   using translatorType = std::unordered_map<unsigned long, unsigned long>;
//   psSmartPointer<psDomain<CellType, NumericType, D>> domain;
//   psSmartPointer<psProcessModel<NumericType>> model;
//   double processDuration;
//   rayTraceDirection sourceDirection;
//   int accuracy = 1000;

//   rayTraceBoundary convertBoundaryCondition(
//       lsBoundaryConditionEnum<D> originalBoundaryCondition) {
//     switch (originalBoundaryCondition) {
//     case lsBoundaryConditionEnum<D>::REFLECTIVE_BOUNDARY:
//       return rayTraceBoundary::REFLECTIVE;

//     case lsBoundaryConditionEnum<D>::INFINITE_BOUNDARY:
//       return rayTraceBoundary::IGNORE;

//     case lsBoundaryConditionEnum<D>::PERIODIC_BOUNDARY:
//       return rayTraceBoundary::PERIODIC;

//     case lsBoundaryConditionEnum<D>::POS_INFINITE_BOUNDARY:
//       return rayTraceBoundary::IGNORE;

//     case lsBoundaryConditionEnum<D>::NEG_INFINITE_BOUNDARY:
//       return rayTraceBoundary::IGNORE;

//       // default:
//       // throw new
//       // InvalidEnumArgumentException(nameof(originalBoundaryCondition),
//       // (int)originalBoundaryCondition, typeof(lsBoundaryConditionEnum));
//     }
//   }

// public:
//   void apply() {
//     psVelocityField<NumericType> velField;

//     double remainingTime = processDuration;
//     const NumericType gridDelta =
//         domain->getLevelSets.back().getGrid().getGridDelta();

//     auto diskMesh = lsSmartPointer<lsMesh<NumericType>>::New();
//     auto translator = lsSmartPointer<translatorType>::New();
//     lsToDiskMesh<NumericType, D> meshConverter(domain->getLevelSets().back(),
//                                                diskMesh, translator);
//     for (auto &dom : domain->getLevelSets()) {
//       meshConverter.inserNextLevelSet(dom);
//     }

//     /* --------- Setup for advection ----------- */
//     lsAdvect<NumericType, D> advectionKernel;
//     for (auto &dom : domain->getLevelSets()) {
//       advectionKernel.inserNextLevelSet(dom);
//     }
//     advectionKernel->setVelocityField(velField);

//     /* --------- Setup for ray tracing ----------- */
//     // Map the domain boundary to the ray tracing boundaries
//     rayTraceBoundary rtBC[D];
//     for (unsigned i = 0; i < D; ++i)
//       rtBC[i] = convertBoundaryCondition(
//           domain.getLevelSets.back()->getGrid()->getBoundaryCondition(i));

//     rayTrace<NumericType, D> rayTrace;
//     rayTrace.setSourceDirection(sourceDirection);
//     rayTrace.setNumberOfRaysPerPoint(accuracy);
//     rayTrace.setBoundaryConditions(rtBC);
//     rayTrace.setCalculateFlux(false);

//     // Iterate until coverages converge. How to know when to stop?
//     {
//       meshConverter.apply();
//       auto numPoints = diskMesh->getNodes().size();
//       model->getSurfaceModel()->setCoverages(numPoints, 0.5);
//       auto points = diskMesh->getNodes();
//       auto normals = *diskMesh->getCellData("Normals");
//       auto materialIds = *diskMesh->getCellData("MaterialIds");
//       rayTrace.setGeometry(points, normals, gridDelta);
//       rayTrace.setMaterialIDs(materialIds);

//       rayTrace.setGlobalData(
//           model->getSurfaceModel()
//               ->getCoverages()); // just set as a reference (using shared ptr)

//       std::vector<std::vector<NumericType>> Rates;

//       for (auto &particle : model->getParticleTypes()) {
//         rayTrace.setParticleType(particle);
//         rayTrace.apply();

//         // fill up rates vector with rates from this particle type
//         for (auto &rate : rayTrace.getLocalData().getVectorData()) {
//           Rates.push_back(std::move(rate));
//         }
//       }

//       // TO DO - the ray tracing above is done to bring the process to the
//       // equilibrium. Now that the initial ray tracing is done, what needs to be
//       // done with coverages and fluxes to update the model properly?

//       // numRaysTraced = points.size() * accuracy;
//       // model.setTotalFluxes(totalIonFlux, totalEtchantFlux, totalPolyFlux,
//       // numRaysTraced); model.updateCoverages(ionEnhancedRate, ionpeRate,
//       // polyRate, 					etchanteRate, etchantpeRate); peCoverage =
//       // model.getPECoverage(); pCoverage = model.getPCoverage(); eCoverage =
//       // model.getECoverage();
//     }

//     while (remainingTime > 0.) {
//       std::vector<std::vector<NumericType>> Rates;

//       meshConverter.apply();
//       auto points = diskMesh->getNodes();
//       auto normals = *diskMesh->getCellData("Normals");
//       auto materialIds = *diskMesh->getCellData("MaterialIds");
//       rayTrace.setGeometry(points, normals, gridDelta);
//       rayTrace.setMaterialIDs(materialIds);
//       rayTrace.setGlobalData(
//           model->getSurfaceModel()
//               ->getCoverages()); // just set as a reference (using shared ptr)

//       for (auto &particle : model->getParticleTypes()) {
//         rayTrace.setParticleType(particle);
//         rayTrace.apply();

//         // fill up rates vector with rates from this particle type
//         for (auto &rate : rayTrace.getLocalData().getVectorData()) {
//           Rates.push_back(std::move(rate));
//         }
//       }

//       velField = model->getVelocityField();
//       velField->setVelocities(
//           model->getSurfaceModel()->calculateVelocities(Rates, materialIds));

//       advectionKernel->setAdvectionTime(remainingTime);
//       advectionKernel->apply();
//       remainingTime -= advectionKernel->getAdvectedTime();
//     }
//   }
};

#endif