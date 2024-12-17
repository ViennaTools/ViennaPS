#include <lsAdvect.hpp>
#include <lsToDiskMesh.hpp>
#include <lsToSurfaceMeshRefined.hpp>

#include <geometries/psMakeTrench.hpp>
#include <psDomain.hpp>
#include <psProcess.hpp>

// #include <curtTrace.hpp>
#include <context.hpp>

using namespace viennaps;

int main() {
  omp_set_num_threads(16);
  using NumericType = float;
  constexpr int D = 3;

  Logger::setLogLevel(LogLevel::DEBUG);

  //   const std::array<NumericType, 10> sticking = {0.1f, 0.2f, 0.3f, 0.4f,
  //   0.5f,
  //                                                 0.6f, 0.7f, 0.8f,
  //                                                 0.9f, 1.f};
  const std::array<NumericType, 1> sticking = {1.f};
  const int processSteps = 10;
  const NumericType gridDelta = .1;
  std::ofstream file("GPU_Benchmark.txt");
  file << "Sticking;Meshing;Tracing;Postprocessing;Advection\n";

  gpu::Context context;
  gpu::CreateContext(context);

  // for (int i = 0; i < sticking.size(); i++) {

  auto domain = SmartPointer<Domain<NumericType, D>>::New();
  MakeTrench<NumericType, D>(domain, gridDelta, 10, 5, 5, 5, 0., 0.5, false,
                             true, Material::Si)
      .apply();

  auto diskMesh = SmartPointer<viennals::Mesh<NumericType>>::New();
  viennals::ToDiskMesh<NumericType, D> diskMesher(diskMesh);

  // gpu::Tracer<NumericType, D> tracer(context);
  //   tracer.setNumberOfRaysFixed(20000000); // 20 million rays
  //   tracer.setUseRandomSeeds(false);

  //   viennals::Advect<NumericType, D> advectionKernel;

  //   auto velocityField =
  //       SmartPointer<DefaultVelocityField<NumericType>>::New(2);
  //   auto translationField = SmartPointer<TranslationField<NumericType>>::New(
  //       velocityField, domain->getMaterialMap());
  //   advectionKernel.setVelocityField(translationField);

  //   for (const auto ls : domain->getLevelSets()) {
  //     diskMesher.insertNextLevelSet(ls);
  //     advectionKernel.insertNextLevelSet(ls);
  //   }

  //   auto particle =
  //       std::make_unique<viennaray::DiffuseParticle<NumericType, D>>(
  //           sticking[i], "flux");
  //   tracer.setParticleType(particle);

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