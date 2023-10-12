#include <SF6O2Etching.hpp>
#include <psMakeHole.hpp>
#include <psWriteVisualizationMesh.hpp>

#include <pscuProcess.hpp>
#include <pscuProcessPipelines.hpp>

int main(int argc, char **argv) {

  omp_set_num_threads(12);
  constexpr int D = 3;

  pscuContext context;
  pscuCreateContext(context);

  const float gridDelta = 1.;
  const float extent = 60.;
  const float holeRadius = 15.;
  const float maskHeight = 30.;

  const float time = 12.;
  const float ionFlux = 10.;
  const float etchantFlux = 2000.;
  const float oxygenFlux = 1000.;
  const float ionEnergy = 100.;
  const float ionSigma = 10.;
  const float exponent = 1000.;

  auto domain = psSmartPointer<psDomain<float, D>>::New();
  psMakeHole<float, DIM>(domain, gridDelta, extent, extent, holeRadius,
                         maskHeight, 0.f, 0.f, false, true, psMaterial::Si)
      .apply();

  domain->printSurface("hole_initial.vtp");

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

  pscuProcess<float, DIM> process(context);
  process.setDomain(domain);
  process.setProcessModel(model);
  process.setNumberOfRaysPerPoint(1000);
  process.setMaxCoverageInitIterations(10);
  process.setProcessDuration(time);
  process.setSmoothFlux(2.);
  process.apply();

  domain->printSurface("hole_etched_t12.vtp");
}