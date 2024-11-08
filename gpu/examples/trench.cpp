#include <SF6O2Etching.hpp>
#include <psMakePlane.hpp>
#include <psUtils.hpp>
#include <psWriteVisualizationMesh.hpp>

#include <psProcess.hpp>
#include <pscuProcess.hpp>
#include <pscuProcessPipelines.hpp>

int main(int argc, char **argv) {

  omp_set_num_threads(16);
  constexpr int D = 3;

  pscuContext context;
  pscuCreateContext(context);
  psLogger::setLogLevel(psLogLevel::INTERMEDIATE);

  const float gridDelta = 1.;
  const float extent = 100.;
  const float trenchWidth = 15.;
  const float spacing = 20.;
  const float maskHeight = 40.;

  const float time = 3.;
  const float ionFlux = 1.;
  const float etchantFlux = 1000.;
  const float oxygenFlux = 500.;
  const float ionEnergy = 100.;
  const float ionSigma = 10.;
  const float exponent = 1000.;

  auto domain = psSmartPointer<psDomain<float, D>>::New();

  {
    double extent2 = extent / 2.;
    double bounds[2 * D] = {-extent2, extent2,  -extent2,
                            extent2,  -extent2, extent2};
    typename lsDomain<float, D>::BoundaryType boundaryCons[D] = {
        lsDomain<float, D>::BoundaryType::REFLECTIVE_BOUNDARY,
        lsDomain<float, D>::BoundaryType::REFLECTIVE_BOUNDARY,
        lsDomain<float, D>::BoundaryType::INFINITE_BOUNDARY};

    float origin[D] = {0., 0., maskHeight};
    float normal[D] = {0., 0., 1.};

    auto mask = psSmartPointer<lsDomain<float, D>>::New(bounds, boundaryCons,
                                                        gridDelta);
    lsMakeGeometry<float, D>(
        mask, lsSmartPointer<lsPlane<float, D>>::New(origin, normal))
        .apply();
    normal[2] = -1.;
    origin[2] = 0.;
    auto maskUnder = psSmartPointer<lsDomain<float, D>>::New(
        bounds, boundaryCons, gridDelta);
    lsMakeGeometry<float, D>(
        maskUnder, lsSmartPointer<lsPlane<float, D>>::New(origin, normal))
        .apply();
    lsBooleanOperation<float, D>(mask, maskUnder).apply();

    auto trench = psSmartPointer<lsDomain<float, D>>::New(bounds, boundaryCons,
                                                          gridDelta);
    float minPoint[D] = {-extent2 + spacing, extent2 - spacing - trenchWidth,
                         -gridDelta};
    float maxPoint[D] = {extent2 - spacing, extent2 - spacing,
                         maskHeight + gridDelta};

    lsMakeGeometry<float, D>(
        trench, lsSmartPointer<lsBox<float, D>>::New(minPoint, maxPoint))
        .apply();
    lsBooleanOperation<float, D>(mask, trench,
                                 lsBooleanOperationEnum::RELATIVE_COMPLEMENT)
        .apply();

    minPoint[0] = -extent2 + spacing;
    minPoint[1] = -extent2 - gridDelta;
    maxPoint[0] = -extent2 + spacing + trenchWidth;
    maxPoint[1] = extent2 - spacing - trenchWidth + gridDelta;

    lsMakeGeometry<float, D>(
        trench, lsSmartPointer<lsBox<float, D>>::New(minPoint, maxPoint))
        .apply();
    lsBooleanOperation<float, D>(mask, trench,
                                 lsBooleanOperationEnum::RELATIVE_COMPLEMENT)
        .apply();

    domain->insertNextLevelSetAsMaterial(mask, psMaterial::Mask);
  }

  psMakePlane<float, D>(domain, 0., psMaterial::Si).apply();
  domain->printSurface("trench_initial.vtp");

  curtParticle<float> ion{.name = "ion",
                          .numberOfData = 3,
                          .cosineExponent = exponent,
                          .meanIonEnergy = ionEnergy,
                          .sigmaIonEnergy = ionSigma,
                          .A_O = 3.f};
  ion.dataLabels.push_back("ionSputteringRate");
  ion.dataLabels.push_back("ionEnhancedRate");
  ion.dataLabels.push_back("oxygenSputteringRate");

  curtParticle<float> etchant{.name = "etchant", .numberOfData = 2};
  etchant.dataLabels.push_back("etchantRate");
  etchant.dataLabels.push_back("sticking_e");

  curtParticle<float> oxygen{.name = "oxygen", .numberOfData = 2};
  oxygen.dataLabels.push_back("oxygenRate");
  oxygen.dataLabels.push_back("sticking_o");

  auto surfModel =
      psSmartPointer<SF6O2Implementation::SurfaceModel<float, DIM>>::New(
          ionFlux, etchantFlux, oxygenFlux, -100000);
  auto velField = psSmartPointer<psDefaultVelocityField<float>>::New(2);
  auto model = psSmartPointer<pscuProcessModel<float>>::New();

  model->insertNextParticleType(ion);
  model->insertNextParticleType(etchant);
  model->insertNextParticleType(oxygen);
  model->setSurfaceModel(surfModel);
  model->setVelocityField(velField);
  model->setProcessName("SF6O2Etching");
  model->setPtxCode(embedded_SF6O2_pipeline);

  //   auto model = psSmartPointer<SF6O2Etching<float, D>>::New(
  //       ionFlux, etchantFlux, oxygenFlux, ionEnergy, ionSigma, exponent, 3.);

  pscuProcess<float, D> process(context);
  process.setDomain(domain);
  process.setProcessModel(model);
  process.setNumberOfRaysPerPoint(1000);
  process.setMaxCoverageInitIterations(30);
  process.setProcessDuration(time);
  process.setSmoothFlux(2.);

  psUtils::Timer timer;
  timer.start();
  process.apply();
  timer.finish();

  std::cout << "Process took: " << timer.currentDuration * 1e-9 << " s"
            << std::endl;

  domain->printSurface("trench_etched.vtp");
}