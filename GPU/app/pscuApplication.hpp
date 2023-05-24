#pragma once

#include "context.hpp"

#include "Application.hpp"

#include "pscuDeposition.hpp"
#include "pscuFluorocarbonEtching.hpp"
#include "pscuSF6O2Etching.hpp"

class pscuApplication : public Application<DIM> {
  pscuContext context;

public:
  pscuApplication(int argc, char **argv) : Application(argc, argv) {
    std::cout << "Initializing CUDA and OptiX ... ";
    pscuCreateContext(context);
    std::cout << "success" << std::endl;
  }

protected:
  void runSimpleDeposition(
      psSmartPointer<psDomain<NumericType, DIM>> processGeometry,
      psSmartPointer<ApplicationParameters> processParams) override {
    curtParticle<NumericType> depoParticle{.name = "depoParticle",
                                           .sticking = processParams->sticking,
                                           .cosineExponent =
                                               processParams->cosinePower};
    depoParticle.dataLabels.push_back("depoRate");

    auto surfModel =
        psSmartPointer<DepoSurfaceModel<NumericType>>::New(context);
    auto velField =
        psSmartPointer<SimpleDepositionVelocityField<NumericType>>::New();
    auto model = psSmartPointer<pscuProcessModel<NumericType>>::New();

    model->insertNextParticleType(depoParticle);
    model->setSurfaceModel(surfModel);
    model->setVelocityField(velField);
    model->setProcessName("Deposition");
    model->setPtxCode(embedded_deposition_pipeline);

    pscuProcess<NumericType, DIM> process(context);
    process.setDomain(processGeometry);
    process.setProcessModel(model);
    process.setNumberOfRaysPerPoint(processParams->raysPerPoint);
    process.setProcessDuration(processParams->processTime *
                               processParams->rate / processParams->sticking);
    process.apply();
  }

  void runSF6O2Etching(
      psSmartPointer<psDomain<NumericType, DIM>> processGeometry,
      psSmartPointer<ApplicationParameters> processParams) override {
    curtParticle<NumericType> ion{.name = "ion",
                                  .numberOfData = 3,
                                  .cosineExponent = 100.f,
                                  .meanIonEnergy = processParams->ionEnergy,
                                  .ionRF = processParams->rfBias,
                                  .A_O = processParams->A_O};
    ion.dataLabels.push_back("ionSputteringRate");
    ion.dataLabels.push_back("ionEnhancedRate");
    ion.dataLabels.push_back("oxygenSputteringRate");

    curtParticle<NumericType> etchant{.name = "etchant", .numberOfData = 1};
    etchant.dataLabels.push_back("etchantRate");

    curtParticle<NumericType> oxygen{.name = "oxygen", .numberOfData = 1};
    oxygen.dataLabels.push_back("oxygenRate");

    auto surfModel = psSmartPointer<pscuSF6O2SurfaceModel<NumericType>>::New(
        context, processParams->ionFlux, processParams->etchantFlux,
        processParams->oxygenFlux);
    auto velField = psSmartPointer<psDefaultVelocityField<NumericType>>::New();
    auto model = psSmartPointer<pscuProcessModel<NumericType>>::New();

    model->insertNextParticleType(ion);
    model->insertNextParticleType(etchant);
    model->insertNextParticleType(oxygen);
    model->setSurfaceModel(surfModel);
    model->setVelocityField(velField);
    model->setProcessName("SF6O2Etching");
    model->setPtxCode(embedded_SF6O2_pipeline);

    pscuProcess<NumericType, DIM> process(context);
    process.setDomain(processGeometry);
    process.setProcessModel(model);
    process.setNumberOfRaysPerPoint(processParams->raysPerPoint);
    process.setMaxCoverageInitIterations(10);
    process.setProcessDuration(processParams->processTime);
    process.apply();
  }

  void runFluorocarbonEtching(
      psSmartPointer<psDomain<NumericType, DIM>> processGeometry,
      psSmartPointer<ApplicationParameters> processParams) override {

    // insert additional top layer to capture deposition
    auto depoLayer = psSmartPointer<lsDomain<NumericType, DIM>>::New(
        processGeometry->getLevelSets()->back());
    int depoId = processGeometry->getLevelSets()->size();
    processGeometry->insertNextLevelSet(depoLayer);

    curtParticle<NumericType> ion{.name = "ion",
                                  .numberOfData = 3,
                                  .cosineExponent = 100.f,
                                  .meanIonEnergy = processParams->ionEnergy,
                                  .ionRF = processParams->rfBias,
                                  .A_O = processParams->A_O};
    ion.dataLabels.push_back("ionSputteringFlux");
    ion.dataLabels.push_back("ionEnhancedFlux");
    ion.dataLabels.push_back("ionPolymerFlux");

    curtParticle<NumericType> etchant{.name = "etchant", .numberOfData = 1};
    etchant.dataLabels.push_back("etchantFlux");

    curtParticle<NumericType> polymer{.name = "polymer", .numberOfData = 1};
    polymer.dataLabels.push_back("polyFlux");

    curtParticle<NumericType> etchantPoly{.name = "etchantPoly",
                                          .numberOfData = 1};
    etchantPoly.dataLabels.push_back("etchantPolyFlux");

    auto surfModel =
        psSmartPointer<pscuFluorocarbonSurfaceModel<NumericType>>::New(
            context, processParams->ionFlux, processParams->etchantFlux,
            processParams->oxygenFlux, processParams->maskId, depoId);
    auto velField = psSmartPointer<psDefaultVelocityField<NumericType>>::New();
    auto model = psSmartPointer<pscuProcessModel<NumericType>>::New();

    model->insertNextParticleType(ion);
    model->insertNextParticleType(etchant);
    model->insertNextParticleType(polymer);
    model->insertNextParticleType(etchantPoly);
    model->setSurfaceModel(surfModel);
    model->setVelocityField(velField);
    model->setProcessName("FluorocarbonEtching");
    model->setPtxCode(embedded_Fluorocarbon_pipeline);

    pscuProcess<NumericType, DIM> process(context);
    process.setDomain(processGeometry);
    process.setProcessModel(model);
    process.setNumberOfRaysPerPoint(processParams->raysPerPoint);
    process.setMaxCoverageInitIterations(10);
    process.setProcessDuration(processParams->processTime);
    process.apply();
  }
};