#include <lsAdvect.hpp>
#include <lsToDiskMesh.hpp>
#include <lsToSurfaceMeshRefined.hpp>

#include <geometries/psMakeTrench.hpp>
#include <psDomain.hpp>
#include <psProcess.hpp>

#include <curtTrace.hpp>
#include <gpu/vcContext.hpp>

using namespace viennaps;

int main() {
  omp_set_num_threads(16);
  using NumericType = float;
  constexpr int D = 3;

  viennacore::Logger::setLogLevel(viennacore::LogLevel::DEBUG);

  //   const std::array<NumericType, 10> sticking = {0.1f, 0.2f, 0.3f, 0.4f,
  //   0.5f,
  //                                                 0.6f, 0.7f, 0.8f,
  //                                                 0.9f, 1.f};
  const std::array<NumericType, 1> sticking = {1.f};
  const int processSteps = 10;
  const NumericType gridDelta = .1;
  std::ofstream file("GPU_Benchmark.txt");
  file << "Sticking;Meshing;Tracing;Postprocessing;Advection\n";

  Context context;
  CreateContext(context);

  // for (int i = 0; i < sticking.size(); i++) {

  auto domain = SmartPointer<Domain<NumericType, D>>::New();
  MakeTrench<NumericType, D>(domain, gridDelta, 10, 5, 5, 5, 0., 0.5, false,
                             true, Material::Si)
      .apply();

  auto diskMesh = SmartPointer<viennals::Mesh<NumericType>>::New();
  viennals::ToDiskMesh<NumericType, D> diskMesher(diskMesh);

  auto surfMesh = SmartPointer<viennals::Mesh<float>>::New();
  viennals::ToSurfaceMeshRefined<NumericType, float, D> surfMesher(
      domain->getLevelSets().back(), surfMesh);

  gpu::Trace<NumericType, D> tracer(context);
  // tracer.setNumberOfRaysFixed(50000000); // 20 million rays
  tracer.setNumberOfRaysPerPoint(3000);
  tracer.setUseRandomSeeds(false);

  viennals::Advect<NumericType, D> advectionKernel;

  auto velocityField = SmartPointer<DefaultVelocityField<NumericType>>::New(2);
  auto translationField = SmartPointer<TranslationField<NumericType>>::New(
      velocityField, domain->getMaterialMap());
  advectionKernel.setVelocityField(translationField);

  for (const auto ls : domain->getLevelSets()) {
    diskMesher.insertNextLevelSet(ls);
    advectionKernel.insertNextLevelSet(ls);
  }

  surfMesher.apply();
  gpu::TriangleMesh mesh;
  mesh.gridDelta = gridDelta;
  mesh.vertices = surfMesh->nodes;
  mesh.triangles = surfMesh->triangles;
  mesh.minimumExtent = surfMesh->minimumExtent;
  mesh.maximumExtent = surfMesh->maximumExtent;

  tracer.setGeometry(mesh);
  tracer.setPipeline("SingleParticlePipeline");

  auto particle = gpu::Particle<NumericType>();
  particle.sticking = 1.f;
  particle.name = "SingleParticle";
  particle.dataLabels.push_back("flux");

  tracer.insertNextParticle(particle);
  tracer.prepareParticlePrograms();
  tracer.apply();

  std::vector<NumericType> flux;
  flux.resize(surfMesh->triangles.size());
  tracer.getFlux(flux.data(), 0, 0);
  surfMesh->getCellData().insertNextScalarData(flux, "flux");

  viennals::VTKWriter<NumericType>(surfMesh, "SingleParticleFlux.vtp").apply();
  //   for (int j = 0; j < processSteps; j++) {
  //     file << sticking[i] << ";";

  //     Timer timer;
  //     timer.start();
  //     diskMesher.apply();
  //     translationField->buildKdTree(mesh->nodes);
  //     timer.finish();
  //     file << timer.currentDuration << ";";

  //     const auto &materialIds =
  //         *mesh->getCellData().getScalarData("MaterialIds");
  //     const auto &normals = *mesh->getCellData().getVectorData("Normals");
  //     tracer.setGeometry(mesh->nodes, normals, gridDelta);
  //     tracer.setMaterialIds(materialIds);

  //     timer.start();
  //     tracer.apply();
  //     timer.finish();
  //     file << timer.currentDuration << ";";

  //     timer.start();
  //     auto &flux = tracer.getLocalData().getVectorData("flux");
  //     tracer.normalizeFlux(flux);
  //     tracer.smoothFlux(flux);
  //     auto velocities =
  //         SmartPointer<std::vector<NumericType>>::New(std::move(flux));
  //     velocityField->setVelocities(velocities);
  //     timer.finish();
  //     file << timer.currentDuration << ";";

  //     timer.start();
  //     advectionKernel.apply();
  //     timer.finish();
  //     file << timer.currentDuration << "\n";
  //   }
  // }

  // file.close();
}