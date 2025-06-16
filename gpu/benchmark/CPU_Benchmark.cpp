#include <lsAdvect.hpp>
#include <lsToDiskMesh.hpp>

#include <psProcess.hpp>
#include <rayTrace.hpp>

#include "BenchmarkGeometry.hpp"

using namespace viennaps;

int main() {
  omp_set_num_threads(16);
  using NumericType = float;
  constexpr int D = DIM;

  const std::array<NumericType, 10> sticking = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f,
                                                0.6f, 0.7f, 0.8f, 0.9f, 1.f};
  // const std::array<NumericType, 1> sticking = {1.f};
  const int processSteps = 10;
  std::ofstream file("CPU_Benchmark.txt");
  file << "Sticking;Meshing;Tracing;Postprocessing;Advection\n";

  for (int i = 0; i < sticking.size(); i++) {

    auto domain = MAKE_GEO<NumericType>();

    auto mesh = viennals::Mesh<NumericType>::New();
    viennals::ToDiskMesh<NumericType, D> mesher(mesh);

    viennaray::Trace<NumericType, D> tracer;
    tracer.setNumberOfRaysPerPoint(3000);
    tracer.setUseRandomSeeds(false);

    viennals::Advect<NumericType, D> advectionKernel;

    auto velocityField =
        SmartPointer<DefaultVelocityField<NumericType, D>>::New(2);
    auto translationField = SmartPointer<TranslationField<NumericType, D>>::New(
        velocityField, domain->getMaterialMap());
    advectionKernel.setVelocityField(translationField);

    for (const auto ls : domain->getLevelSets()) {
      mesher.insertNextLevelSet(ls);
      advectionKernel.insertNextLevelSet(ls);
    }

    auto particle =
        std::make_unique<viennaray::DiffuseParticle<NumericType, D>>(
            sticking[i], "flux");
    tracer.setParticleType(particle);

    for (int j = 0; j < processSteps; j++) {
      file << sticking[i] << ";";

      Timer timer;
      timer.start();
      mesher.apply();
      translationField->buildKdTree(mesh->nodes);
      timer.finish();
      file << timer.currentDuration << ";";

      const auto &materialIds =
          *mesh->getCellData().getScalarData("MaterialIds");
      const auto &normals = *mesh->getCellData().getVectorData("Normals");
      tracer.setGeometry(mesh->nodes, normals, GRID_DELTA);
      tracer.setMaterialIds(materialIds);

      timer.start();
      tracer.apply();
      timer.finish();
      file << timer.currentDuration << ";";

      timer.start();
      auto &flux = tracer.getLocalData().getVectorData("flux");
      tracer.normalizeFlux(flux);
      tracer.smoothFlux(flux);
      auto velocities =
          SmartPointer<std::vector<NumericType>>::New(std::move(flux));
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