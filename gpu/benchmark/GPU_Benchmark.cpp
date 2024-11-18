#include <context.hpp>

#include <curtTracer.hpp>
#include <lsToSurfaceMeshRefined.hpp>
#include <pscuDeposition.hpp>

#include <psMakeTrench.hpp>
#include <psUtils.hpp>

using namespace viennaps;

int main() {
  using NumericType = float;
  constexpr int D = 3;

  auto domain = SmartPointer<Domain<NumericType, D>>::New();
  MakeTrench<NumericType, D>(domain, .1, 10, 5, 5, 5, 0., 0.5, false, true,
                             Material::Metal)
      .apply();

  auto tree =
      SmartPointer<KDTree<NumericType, std::array<NumericType, 3>>>::New();

  {
    int i = 0;
    for (auto ls : domain->getLevelSets()) {
      domain->saveLevelSetMesh("mesh_" + std::to_string(i) + ".vtp");
      domain->saveSurfaceMesh("surfmesh_" + std::to_string(i) + ".vtp");
      ++i;
    }
  }

  {
    auto mesh = SmartPointer<viennals::Mesh<NumericType>>::New();
    viennals::ToSurfaceMeshRefined<NumericType, NumericType, D> mesher(
        domain, mesh, tree);

    Timer timer;
    timer.start();
    mesher.apply();
    timer.finish();

    std::cout << "Culs time: " << timer.currentDuration * 1e-6 << " ms"
              << std::endl;

    viennals::VTKWriter<NumericType>(mesh, "culs_surface.vtp").apply();
  }

  {
    auto mesh = SmartPointer<viennals::Mesh<NumericType>>::New();
    psUtils::Timer timer;
    timer.start();
    viennals::ToSurfaceMesh<NumericType, D>(domain->getLevelSets()->back(),
                                            mesh)
        .apply();
    timer.finish();

    std::cout << "LS time: " << timer.currentDuration * 1e-6 << " ms"
              << std::endl;

    viennals::VTKWriter<NumericType>(mesh, "surface.vtp").apply();
  }

  auto vmesh = SmartPointer<viennals::Mesh<NumericType>>::New();
  viennals::ToVoxelMesh<NumericType, D> voxelMesh;
  voxelMesh.setMesh(vmesh);
  for (auto ls : domain->getLevelSets()) {
    voxelMesh.insertNextLevelSet(ls);
  }
  voxelMesh.apply();
  viennals::VTKWriter<NumericType>(vmesh, "voxel.vtu").apply();

  // pscuContext context;
  // pscuCreateContext(context);

  // auto geometry = SmartPointer<Domain<NumericType, DIM>>::New();
  // MakeTrench<NumericType, DIM>(geometry, 1., 150., 200., 50., 300., 0., 0.)
  //     .apply();
  // geometry->printSurface("initial.vtp");

  // curtTracer<NumericType, DIM> tracer(context);
  // tracer.setLevelSet(geometry->getLevelSets().back());
  // tracer.setNumberOfRaysPerPoint(10000);
  // tracer.setPipeline(embedded_deposition_pipeline);

  // std::array<NumericType, 10> sticking = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f,
  //                                         0.6f, 0.7f, 0.8f, 0.9f, 1.f};
  // curtParticle<NumericType> particle{
  //     .name = "depoParticle", .sticking = 1.f, .cosineExponent = 1};
  // tracer.insertNextParticle(particle);
  // tracer.prepareParticlePrograms();

  // std::ofstream file("BenchmarkResults.txt");

  // psUtils::Timer timer;
  // for (int i = 0; i < 10; i++) {
  //   auto &particle = tracer.getParticles()[0];
  //   particle.sticking = sticking[i];

  //   file << sticking[i] << " ";
  //   for (int j = 0; j < 10; j++) {
  //     timer.start();
  //     tracer.apply();
  //     timer.finish();
  //     file << timer.currentDuration << " ";
  //   }

  //   file << "\n";
  // }

  // file.close();
}