#include <lsToDiskMesh.hpp>
#include <lsToSurfaceMeshRefined.hpp>

#include <models/psSingleParticleProcess.hpp>
#include <psProcess.hpp>

#include <curtTrace.hpp>
#include <gpu/vcContext.hpp>
#include <utElementToPointData.hpp>

#include "BenchmarkGeometry.hpp"

using namespace viennaps;

int main() {
  omp_set_num_threads(16);
  using NumericType = float;
  constexpr int D = 3;

  viennacore::Logger::setLogLevel(viennacore::LogLevel::DEBUG);

  const NumericType sticking = 0.1f;

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
      domain->getSurface(), surfMesh, elementKdTree);

  gpu::Trace<NumericType, D> tracer(context);
  tracer.setNumberOfRaysPerPoint(3000);
  tracer.setUseRandomSeeds(false);

  diskMesher.apply();
  surfMesher.apply();
  gpu::TriangleMesh mesh;
  mesh.gridDelta = GRID_DELTA;
  mesh.vertices = surfMesh->nodes;
  mesh.triangles = surfMesh->triangles;
  mesh.minimumExtent = surfMesh->minimumExtent;
  mesh.maximumExtent = surfMesh->maximumExtent;

  tracer.setGeometry(mesh);
  tracer.setPipeline("SingleParticlePipeline", context->modulePath);

  auto particle = gpu::Particle<NumericType>();
  particle.sticking = sticking;
  particle.name = "SingleParticle";
  particle.dataLabels.push_back("flux");

  tracer.insertNextParticle(particle);
  tracer.prepareParticlePrograms();
  tracer.apply();

  auto pointData = SmartPointer<viennals::PointData<NumericType>>::New();
  gpu::ElementToPointData<NumericType>(tracer.getResults(), pointData,
                                       tracer.getParticles(), elementKdTree,
                                       diskMesh, surfMesh, GRID_DELTA)
      .apply();

  tracer.downloadResultsToPointData(surfMesh->getCellData());
  viennals::VTKWriter<NumericType>(surfMesh, "GPU_SingleParticleFlux_tri.vtp")
      .apply();
  diskMesh->cellData = *pointData;
  viennals::VTKWriter<NumericType>(diskMesh, "GPU_SingleParticleFlux_disk.vtp")
      .apply();

  // CPU
  auto model =
      SmartPointer<SingleParticleProcess<NumericType, D>>::New(1., sticking);
  Process<NumericType, D> process(domain, model);

  auto flux = process.calculateFlux();

  viennals::VTKWriter<NumericType>(flux, "CPU_SingleParticleFlux_disk.vtp")
      .apply();

  std::ofstream file_top("compare_flux_top.txt");
  std::ofstream file_bottom("compare_flux_bottom.txt");
  std::ofstream file_side_1("compare_flux_side_1.txt");
  std::ofstream file_side_2("compare_flux_side_2.txt");
  file_top << "x;y;CPU;GPU\n";
  file_bottom << "x;y;CPU;GPU\n";
  file_side_1 << "x;y;CPU;GPU\n";
  file_side_2 << "x;y;CPU;GPU\n";

  const auto numNodes = flux->nodes.size();

  auto cpu_flxu = flux->getCellData().getScalarData("particleFlux");
  auto gpu_flxu = diskMesh->getCellData().getScalarData("flux");

  // top points
  const float top = 25.2f;
  const float bottom = 0.25f;

  for (std::size_t i = 0; i < numNodes; i++) {
    auto &node = flux->nodes[i];
    if (std::abs(node[2] - top) < 0.1 && std::abs(node[1]) < 0.01) {
      file_top << node[0] << ";" << node[1] << ";" << cpu_flxu->at(i) << ";"
               << gpu_flxu->at(i) << "\n";
    }

    if (std::abs(node[2] - bottom) < 0.1 && std::abs(node[1]) < 0.01) {
      file_bottom << node[0] << ";" << node[1] << ";" << cpu_flxu->at(i) << ";"
                  << gpu_flxu->at(i) << "\n";
    }

    if (std::abs(node[0] - 2.44) < 0.025 && std::abs(node[1]) < 0.01) {
      file_side_1 << node[1] << ";" << node[2] << ";" << cpu_flxu->at(i) << ";"
                  << gpu_flxu->at(i) << "\n";
    }

    if (std::abs(node[0] + 2.5) < 0.025 && std::abs(node[1]) < 0.01) {
      file_side_2 << node[1] << ";" << node[2] << ";" << cpu_flxu->at(i) << ";"
                  << gpu_flxu->at(i) << "\n";
    }
  }
}