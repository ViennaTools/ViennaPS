#include <lsToDiskMesh.hpp>
#include <lsToSurfaceMeshRefined.hpp>

#include <curtTrace.hpp>
#include <gpu/vcContext.hpp>
#include <utElementToPointData.hpp>

#include "BenchmarkGeometry.hpp"

using namespace viennaps;

int main() {
  omp_set_num_threads(16);
  using NumericType = float;
  constexpr int D = DIM;

  viennacore::Logger::setLogLevel(viennacore::LogLevel::DEBUG);

  const NumericType sticking = 0.1f;
  std::ofstream file("GPU_Benchmark_single.txt");
  file << "Sticking;Meshing;Tracing;Postprocessing\n";
  file << sticking << ";";

  Context context;
  CreateContext(context);

  auto domain = MAKE_GEO<NumericType>();

  auto diskMesh = SmartPointer<viennals::Mesh<NumericType>>::New();
  viennals::ToDiskMesh<NumericType, D> diskMesher(diskMesh);
  for (const auto ls : domain->getLevelSets()) {
    diskMesher.insertNextLevelSet(ls);
  }

  auto elementKdTree = SmartPointer<KDTree<float, Vec3Df>>::New();
  auto surfMesh = SmartPointer<viennals::Mesh<float>>::New();
  viennals::ToSurfaceMeshRefined<NumericType, float, D> surfMesher(
      domain->getLevelSets().back(), surfMesh, elementKdTree);

  gpu::Trace<NumericType, D> tracer(context);
  tracer.setNumberOfRaysPerPoint(5000);
  tracer.setUseRandomSeeds(false);

  Timer timer;
  timer.start();
  diskMesher.apply();
  surfMesher.apply();
  gpu::TriangleMesh mesh;
  mesh.gridDelta = GRID_DELTA;
  mesh.vertices = surfMesh->nodes;
  mesh.triangles = surfMesh->triangles;
  mesh.minimumExtent = surfMesh->minimumExtent;
  mesh.maximumExtent = surfMesh->maximumExtent;
  timer.finish();
  file << timer.currentDuration << ";";
  std::cout << "Meshing time: " << timer.currentDuration * 1e-6 << " ms"
            << std::endl;

  tracer.setGeometry(mesh);
  tracer.setPipeline("SingleParticlePipeline");

  auto particle = gpu::Particle<NumericType>();
  particle.sticking = sticking;
  particle.name = "SingleParticle";
  particle.dataLabels.push_back("flux");

  tracer.insertNextParticle(particle);
  tracer.prepareParticlePrograms();

  timer.start();
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
  auto pointData = SmartPointer<viennals::PointData<NumericType>>::New();
  gpu::ElementToPointData<NumericType>(tracer.getResults(), pointData,
                                       tracer.getParticles(), elementKdTree,
                                       diskMesh, surfMesh, GRID_DELTA)
      .apply();
  timer.finish();
  file << timer.currentDuration << "\n";
  std::cout << "Postprocessing time: " << timer.currentDuration * 1e-6 << " ms"
            << std::endl;

  viennals::VTKWriter<NumericType>(surfMesh, "GPU_SingleParticleFlux_tri.vtp")
      .apply();
  diskMesh->cellData = *pointData;
  viennals::VTKWriter<NumericType>(diskMesh, "GPU_SingleParticleFlux_disk.vtp")
      .apply();

  file.close();
}