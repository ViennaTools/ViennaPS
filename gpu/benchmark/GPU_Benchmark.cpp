#include <lsAdvect.hpp>
#include <lsToDiskMesh.hpp>

#include <psProcess.hpp>

#include <curtMesh.hpp>
#include <curtTrace.hpp>
#include <gpu/vcContext.hpp>
#include <utElementToPointData.hpp>

#include "BenchmarkGeometry.hpp"

using namespace viennaps;

int main() {
  omp_set_num_threads(16);
  using NumericType = float;
  constexpr int D = DIM;

  const std::array<NumericType, 10> sticking = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f,
                                                0.6f, 0.7f, 0.8f, 0.9f, 1.f};
  //   const std::array<NumericType, 1> sticking = {1.f};
  const int processSteps = 10;
  std::ofstream file("GPU_Benchmark.txt");
  file << "Sticking;Meshing;Tracing;Postprocessing;Advection\n";

  Context context;
  context.create();

  gpu::Trace<NumericType, D> tracer(context);
  tracer.setNumberOfRaysPerPoint(3000);
  tracer.setUseRandomSeeds(false);
  tracer.setPipeline("SingleParticlePipeline", context.modulePath);

  auto particle = gpu::Particle<NumericType>();
  particle.name = "SingleParticle";
  particle.dataLabels.push_back("flux");
  tracer.insertNextParticle(particle);
  tracer.prepareParticlePrograms();

  for (int i = 0; i < sticking.size(); i++) {
    auto domain = MAKE_GEO<NumericType>();

    auto diskMesh = SmartPointer<viennals::Mesh<NumericType>>::New();
    viennals::ToDiskMesh<NumericType, D> diskMesher(diskMesh);

    auto elementKdTree = SmartPointer<KDTree<float, Vec3Df>>::New();
    auto surfMesh = SmartPointer<viennals::Mesh<float>>::New();
    gpu::CreateSurfaceMesh<NumericType, float, D> surfMesher(
        domain->getLevelSets().back(), surfMesh, elementKdTree);

    viennals::Advect<NumericType, D> advectionKernel;

    auto velocityField =
        SmartPointer<DefaultVelocityField<NumericType, D>>::New(2);
    auto translationField = SmartPointer<TranslationField<NumericType, D>>::New(
        velocityField, domain->getMaterialMap());
    advectionKernel.setVelocityField(translationField);

    for (const auto ls : domain->getLevelSets()) {
      diskMesher.insertNextLevelSet(ls);
      advectionKernel.insertNextLevelSet(ls);
    }

    auto &particles = tracer.getParticles();
    particles[0].sticking = sticking[i];

    for (int j = 0; j < processSteps; j++) {
      file << sticking[i] << ";";

      Timer timer;
      timer.start();
      diskMesher.apply();
      surfMesher.apply();
      gpu::TriangleMesh<float> mesh(GRID_DELTA, surfMesh);
      translationField->buildKdTree(diskMesh->nodes);
      timer.finish();
      file << timer.currentDuration << ";";

      tracer.setGeometry(mesh);

      timer.start();
      tracer.apply();
      timer.finish();
      file << timer.currentDuration << ";";

      timer.start();
      auto pointData = SmartPointer<viennals::PointData<NumericType>>::New();
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