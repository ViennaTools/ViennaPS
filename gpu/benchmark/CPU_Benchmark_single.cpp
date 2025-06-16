#include <lsAdvect.hpp>
#include <lsToDiskMesh.hpp>

#include <psProcess.hpp>
#include <rayTrace.hpp>

#include "BenchmarkGeometry.hpp"

using namespace viennaps;

int main() {
  omp_set_num_threads(1);
  using NumericType = float;
  constexpr int D = DIM;

  int numRuns = 1;
  const NumericType sticking = 0.5f;
  std::ofstream file("CPU_Benchmark_single.txt");
  file << "Sticking;Meshing;Tracing;Postprocessing;NumberOfTraces\n";

  auto domain = MAKE_GEO<NumericType>();

  auto mesh = viennals::Mesh<NumericType>::New();
  viennals::ToDiskMesh<NumericType, D> mesher(mesh);

  viennaray::Trace<NumericType, D> tracer;
  tracer.setNumberOfRaysPerPoint(3000);
  tracer.setUseRandomSeeds(false);

  for (const auto ls : domain->getLevelSets()) {
    mesher.insertNextLevelSet(ls);
  }

  auto particle = std::make_unique<viennaray::DiffuseParticle<NumericType, D>>(
      sticking, "flux");
  tracer.setParticleType(particle);

  for (int i = 0; i < numRuns; i++) {
    file << sticking << ";";

    Timer timer;
    timer.start();
    mesher.apply();
    timer.finish();
    file << timer.currentDuration << ";";
    std::cout << "Meshing time: " << timer.currentDuration * 1e-6 << " ms"
              << std::endl;

    const auto &materialIds = *mesh->getCellData().getScalarData("MaterialIds");
    const auto &normals = *mesh->getCellData().getVectorData("Normals");
    tracer.setGeometry(mesh->nodes, normals, GRID_DELTA);
    tracer.setMaterialIds(materialIds);

    timer.start();
    tracer.apply();
    timer.finish();
    file << timer.currentDuration << ";";
    std::cout << "Trace time: " << timer.currentDuration * 1e-6 << " ms"
              << std::endl;

    auto info = tracer.getRayTraceInfo();

    timer.start();
    auto &flux = tracer.getLocalData().getVectorData("flux");
    tracer.normalizeFlux(flux);
    // tracer.smoothFlux(flux);
    timer.finish();
    file << timer.currentDuration << ";";
    std::cout << "Postprocessing time: " << timer.currentDuration * 1e-6
              << " ms" << std::endl;

    file << info.totalRaysTraced << "\n";

    mesh->getCellData().insertNextScalarData(flux, "flux");
    viennals::VTKWriter<NumericType>(mesh, "CPU_SingleParticleFlux_disk.vtp")
        .apply();
  }

  file.close();
}