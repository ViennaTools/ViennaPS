#pragma once

#include "context.hpp"

#include "application.hpp"

// #include <SF6O2Etching.hpp>

#include <pscuProcess.hpp>
#include <pscuProcessPipelines.hpp>

namespace viennaps {

class gpuApplication : public Application<DIM> {
  gpuContext context;

public:
  gpuApplication(int argc, char **argv) : Application(argc, argv) {
    std::cout << "Initializing CUDA and OptiX ... ";
    CreateContext(context);
    std::cout << "success" << std::endl;
  }

protected:
  void runSingleParticleProcess(
      SmartPointer<viennaps::Domain<NumericType, DIM>> processGeometry,
      SmartPointer<ApplicationParameters> processParams) override {

    // copy top layer for deposition
    processGeometry->duplicateTopLevelSet(processParams->material);

    // particle
    curtParticle<NumericType> depoParticle{.name = "depoParticle",
                                           .sticking = processParams->sticking,
                                           .cosineExponent =
                                               processParams->cosinePower};
    depoParticle.dataLabels.push_back("depoRate");

    auto surfModel = SmartPointer<SimpleDepositionImplementation::SurfaceModel<
        NumericType, DIM>>::New(processParams->rate);
    auto velField = SmartPointer<DefaultVelocityField<NumericType>>::New(2);
    auto model = SmartPointer<pscuProcessModel<NumericType>>::New();

    model->insertNextParticleType(depoParticle);
    model->setSurfaceModel(surfModel);
    model->setVelocityField(velField);
    model->setProcessName("SimpleDeposition");
    model->setPtxCode(embedded_deposition_pipeline);

    pscuProcess<NumericType, DIM> process(context);
    process.setDomain(processGeometry);
    process.setProcessModel(model);
    process.setNumberOfRaysPerPoint(processParams->raysPerPoint);
    process.setProcessDuration(processParams->processTime);
    process.apply();
  }

  // void
  // runSF6O2Etching(SmartPointer<psDomain<NumericType, DIM>> processGeometry,
  //                 SmartPointer<ApplicationParameters> processParams) override
  //                 {
  //   curtParticle<NumericType> ion{.name = "ion",
  //                                 .numberOfData = 3,
  //                                 .cosineExponent =
  //                                 processParams->ionExponent, .meanIonEnergy
  //                                 = processParams->ionEnergy, .sigmaIonEnergy
  //                                 =
  //                                     processParams->sigmaIonEnergy,
  //                                 .A_O = processParams->A_O};
  //   ion.dataLabels.push_back("ionSputteringRate");
  //   ion.dataLabels.push_back("ionEnhancedRate");
  //   ion.dataLabels.push_back("oxygenSputteringRate");

  //   curtParticle<NumericType> etchant{.name = "etchant", .numberOfData = 2};
  //   etchant.dataLabels.push_back("etchantRate");
  //   etchant.dataLabels.push_back("sticking_e");

  //   curtParticle<NumericType> oxygen{.name = "oxygen", .numberOfData = 2};
  //   oxygen.dataLabels.push_back("oxygenRate");
  //   oxygen.dataLabels.push_back("sticking_o");

  //   auto surfModel =
  //       SmartPointer<SF6O2Implementation::SurfaceModel<NumericType,
  //       DIM>>::New(
  //           processParams->ionFlux, processParams->etchantFlux,
  //           processParams->oxygenFlux, processParams->etchStopDepth);
  //   auto velField =
  //   SmartPointer<psDefaultVelocityField<NumericType>>::New(2); auto model =
  //   SmartPointer<pscuProcessModel<NumericType>>::New();

  //   model->insertNextParticleType(ion);
  //   model->insertNextParticleType(etchant);
  //   model->insertNextParticleType(oxygen);
  //   model->setSurfaceModel(surfModel);
  //   model->setVelocityField(velField);
  //   model->setProcessName("SF6O2Etching");
  //   model->setPtxCode(embedded_SF6O2_pipeline);

  //   pscuProcess<NumericType, DIM> process(context);
  //   process.setDomain(processGeometry);
  //   process.setProcessModel(model);
  //   process.setNumberOfRaysPerPoint(processParams->raysPerPoint);
  //   process.setMaxCoverageInitIterations(10);
  //   process.setProcessDuration(processParams->processTime);
  //   process.setSmoothFlux(processParams->smoothFlux);
  //   process.apply();
  // }

  // void runFluorocarbonEtching(
  //     SmartPointer<psDomain<NumericType, DIM>> processGeometry,
  //     SmartPointer<ApplicationParameters> processParams) override {

  //   // insert additional top layer to capture deposition
  //   processGeometry->duplicateTopLevelSet(psMaterial::Polymer);

  //   curtParticle<NumericType> ion{.name = "ion",
  //                                 .numberOfData = 3,
  //                                 .cosineExponent = 100.f,
  //                                 .meanIonEnergy = processParams->ionEnergy,
  //                                 .ionRF = processParams->rfBias,
  //                                 .A_O = processParams->A_O};
  //   ion.dataLabels.push_back("ionSputteringFlux");
  //   ion.dataLabels.push_back("ionEnhancedFlux");
  //   ion.dataLabels.push_back("ionPolymerFlux");

  //   curtParticle<NumericType> etchant{.name = "etchant", .numberOfData = 1};
  //   etchant.dataLabels.push_back("etchantFlux");

  //   curtParticle<NumericType> polymer{.name = "polymer", .numberOfData = 1};
  //   polymer.dataLabels.push_back("polyFlux");

  //   curtParticle<NumericType> etchantPoly{.name = "etchantPoly",
  //                                         .numberOfData = 1};
  //   etchantPoly.dataLabels.push_back("etchantPolyFlux");

  //   auto surfModel =
  //       SmartPointer<pscuFluorocarbonSurfaceModel<NumericType>>::New(
  //           context, processParams->ionFlux, processParams->etchantFlux,
  //           processParams->oxygenFlux, processParams->temperature);
  //   auto velField =
  //   SmartPointer<psDefaultVelocityField<NumericType>>::New(2); auto model =
  //   SmartPointer<pscuProcessModel<NumericType>>::New();

  //   model->insertNextParticleType(ion);
  //   model->insertNextParticleType(etchant);
  //   model->insertNextParticleType(polymer);
  //   model->insertNextParticleType(etchantPoly);
  //   model->setSurfaceModel(surfModel);
  //   model->setVelocityField(velField);
  //   model->setProcessName("FluorocarbonEtching");
  //   model->setPtxCode(embedded_Fluorocarbon_pipeline);

  //   pscuProcess<NumericType, DIM> process(context);
  //   process.setDomain(processGeometry);
  //   process.setProcessModel(model);
  //   process.setNumberOfRaysPerPoint(processParams->raysPerPoint);
  //   process.setMaxCoverageInitIterations(10);
  //   process.setProcessDuration(processParams->processTime);
  //   process.apply();
  // }
};

} // namespace viennaps