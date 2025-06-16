#include <lsToDiskMesh.hpp>

#include <psgCreateSurfaceMesh.hpp>
#include <psgElementToPointData.hpp>
#include <psgPointToElementData.hpp>
#include <raygTrace.hpp>
#include <vcContext.hpp>

#include "BenchmarkGeometry.hpp"

using namespace viennaps;

// #define COUNT_RAYS

int main() {
  omp_set_num_threads(1);
  using NumericType = float;
  constexpr int D = DIM;

  viennacore::Logger::setLogLevel(viennacore::LogLevel::WARNING);

  constexpr int numRuns = 10;
  constexpr NumericType sticking = 0.1f;
#ifdef COUNT_RAYS
  std::ofstream file("GPU_Benchmark_single_with_ray_count.txt");
#else
  std::ofstream file("GPU_Benchmark_single_no_ray_count.txt");
#endif
  file << "Sticking;Meshing;Tracing;Postprocessing";
#ifdef COUNT_RAYS
  file << ";NumberOfTraces\n";
#else
  file << "\n";
#endif

  Context context;
  context.create();

  auto domain = MAKE_GEO<NumericType>();

  auto diskMesh = viennals::Mesh<NumericType>::New();
  viennals::ToDiskMesh<NumericType, D> diskMesher(diskMesh);
  for (const auto &ls : domain->getLevelSets()) {
    diskMesher.insertNextLevelSet(ls);
  }

  auto elementKdTree = SmartPointer<KDTree<float, Vec3Df>>::New();
  auto surfMesh = viennals::Mesh<float>::New();
  gpu::CreateSurfaceMesh<NumericType, float, D> surfMesher(
      domain->getSurface(), surfMesh, elementKdTree);

  for (int i = 0; i < numRuns; i++) {
    file << sticking << ";";

    viennaray::gpu::Trace<NumericType, D> tracer(context);
    tracer.setNumberOfRaysPerPoint(3000);
    tracer.setUseRandomSeeds(false);

    Timer timer;
    timer.start();
    diskMesher.apply();
    surfMesher.apply();
    auto mesh = gpu::CreateTriangleMesh(GRID_DELTA, surfMesh);
    timer.finish();
    file << timer.currentDuration << ";";
    std::cout << "Meshing time: " << timer.currentDuration * 1e-6 << " ms"
              << std::endl;

    tracer.setPipeline("SingleParticlePipeline", context.modulePath);
#ifdef COUNT_RAYS
    int rayCount = 0;
    tracer.setParameters(rayCount);
#endif

    auto particle = viennaray::gpu::Particle<NumericType>();
    particle.sticking = sticking;
    particle.name = "SingleParticle";
    particle.dataLabels.emplace_back("flux");

    tracer.insertNextParticle(particle);
    tracer.prepareParticlePrograms();

    timer.start();
    tracer.setGeometry(mesh); // include GAS build
    tracer.apply();
    timer.finish();
    file << timer.currentDuration << ";";
    std::cout << "Trace time: " << timer.currentDuration * 1e-6 << " ms"
              << std::endl;

    timer.start();
    std::vector<NumericType> flux;
    flux.resize(surfMesh->triangles.size());
    tracer.getFlux(flux.data(), 0, 0);
    surfMesh->getCellData().insertNextScalarData(flux, "flux");
    auto pointData = viennals::PointData<NumericType>::New();
    gpu::ElementToPointData<NumericType>(tracer.getResults(), pointData,
                                         tracer.getParticles(), elementKdTree,
                                         diskMesh, surfMesh, GRID_DELTA)
        .apply();
    timer.finish();
    file << timer.currentDuration;
    std::cout << "Postprocessing time: " << timer.currentDuration * 1e-6
              << " ms" << std::endl;

    if (i == numRuns - 1) {
      KDTree<NumericType, Vec3Df> pointKdTree;
      pointKdTree.setPoints(diskMesh->nodes);
      pointKdTree.build();
      gpu::CudaBuffer dummy;
      gpu::PointToElementData<NumericType>(dummy, pointData, pointKdTree,
                                           surfMesh, true, false)
          .apply();
      viennals::VTKWriter<NumericType>(surfMesh,
                                       "GPU_SingleParticleFlux_tri.vtp")
          .apply();
      diskMesh->cellData = *pointData;
      viennals::VTKWriter<NumericType>(diskMesh,
                                       "GPU_SingleParticleFlux_disk.vtp")
          .apply();
    }

#ifdef COUNT_RAYS
    auto &buffer = tracer.getParameterBuffer();
    buffer.download(&rayCount, 1);
    std::cout << "Number of rays: " << rayCount << std::endl;
    file << ";" << rayCount << "\n";
#else
    file << "\n";
#endif
  }

  file.close();
}