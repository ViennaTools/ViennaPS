#include <lsAdvect.hpp>
#include <lsToDiskMesh.hpp>

#include <psProcess.hpp>

#include <psgCreateSurfaceMesh.hpp>
#include <psgElementToPointData.hpp>
#include <raygMesh.hpp>
#include <raygTrace.hpp>
#include <vcContext.hpp>

#include "BenchmarkGeometry.hpp"

using namespace viennaps;

int main() {
  omp_set_num_threads(16);
  using NumericType = float;
  constexpr int D = DIM;

  constexpr std::array<NumericType, 10> sticking = {
      0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.f};
  //   const std::array<NumericType, 1> sticking = {1.f};
  const int processSteps = 10;
  std::ofstream file("GPU_Benchmark.txt");
  file << "Sticking;Meshing;Tracing;Postprocessing;Advection\n";

  Context context;
  context.create();

  viennaray::gpu::Trace<NumericType, D> tracer(context);
  tracer.setNumberOfRaysPerPoint(3000);
  tracer.setUseRandomSeeds(false);
  tracer.setPipeline("SingleParticlePipeline", context.modulePath);

  auto particle = viennaray::gpu::Particle<NumericType>();
  particle.name = "SingleParticle";
  particle.dataLabels.emplace_back("flux");
  tracer.insertNextParticle(particle);
  tracer.prepareParticlePrograms();

  for (auto i : sticking) {
    auto domain = MAKE_GEO<NumericType>();

    auto diskMesh = viennals::Mesh<NumericType>::New();
    viennals::ToDiskMesh<NumericType, D> diskMesher(diskMesh);

    auto elementKdTree = SmartPointer<KDTree<float, Vec3Df>>::New();
    auto surfMesh = viennals::Mesh<float>::New();
    gpu::CreateSurfaceMesh<NumericType, float, D> surfMesher(
        domain->getLevelSets().back(), surfMesh, elementKdTree);

    viennals::Advect<NumericType, D> advectionKernel;

    auto velocityField =
        SmartPointer<DefaultVelocityField<NumericType, D>>::New(2);
    auto translationField = SmartPointer<TranslationField<NumericType, D>>::New(
        velocityField, domain->getMaterialMap());
    advectionKernel.setVelocityField(translationField);

    for (const auto &ls : domain->getLevelSets()) {
      diskMesher.insertNextLevelSet(ls);
      advectionKernel.insertNextLevelSet(ls);
    }

    auto &particles = tracer.getParticles();
    particles[0].sticking = i;

    for (int j = 0; j < processSteps; j++) {
      file << i << ";";

      Timer timer;
      timer.start();
      diskMesher.apply();
      surfMesher.apply();
      auto mesh = gpu::CreateTriangleMesh(GRID_DELTA, surfMesh);
      translationField->buildKdTree(diskMesh->nodes);
      timer.finish();
      file << timer.currentDuration << ";";

      tracer.setGeometry(mesh);

      timer.start();
      tracer.apply();
      timer.finish();
      file << timer.currentDuration << ";";

      timer.start();
      auto pointData = viennals::PointData<NumericType>::New();
      gpu::ElementToPointData<NumericType>(tracer.getResults(), pointData,
                                           tracer.getParticles(), elementKdTree,
                                           diskMesh, surfMesh, GRID_DELTA)
          .apply();
      auto velocities = SmartPointer<std::vector<NumericType>>::New(
          std::move(*pointData->getScalarData("flux")));
      velocityField->prepare(domain, velocities, 0.);
      timer.finish();
      file << timer.currentDuration << ";";

      timer.start();
      advectionKernel.apply();
      timer.finish();
      file << timer.currentDuration << "\n";
    }
  }

  file.close();
}