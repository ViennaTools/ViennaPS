#include <context.hpp>

#include <curtTracer.hpp>
#include <pscuDeposition.hpp>

#include <psMakeTrench.hpp>
#include <psUtils.hpp>

int main() {

  pscuContext context;
  pscuCreateContext(context);

  auto geometry = psSmartPointer<psDomain<NumericType, DIM>>::New();
  psMakeTrench<NumericType, DIM>(geometry, 1., 150., 200., 50., 300., 0., 0.)
      .apply();
  geometry->printSurface("initial.vtp");

  curtTracer<NumericType, DIM> tracer(context);
  tracer.setLevelSet(geometry->getLevelSets()->back());
  tracer.setNumberOfRaysPerPoint(10000);
  tracer.setPipeline(embedded_deposition_pipeline);

  std::array<NumericType, 10> sticking = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f,
                                          0.6f, 0.7f, 0.8f, 0.9f, 1.f};
  curtParticle<NumericType> particle{
      .name = "depoParticle", .sticking = 1.f, .cosineExponent = 1};
  tracer.insertNextParticle(particle);
  tracer.prepareParticlePrograms();

  std::ofstream file("BenchmarkResults.txt");

  psUtils::Timer timer;
  for (int i = 0; i < 10; i++) {
    auto &particle = tracer.getParticles()[0];
    particle.sticking = sticking[i];

    file << sticking[i] << " ";
    for (int j = 0; j < 10; j++) {
      timer.start();
      tracer.apply();
      timer.finish();
      file << timer.currentDuration << " ";
    }

    file << "\n";
  }

  file.close();
}