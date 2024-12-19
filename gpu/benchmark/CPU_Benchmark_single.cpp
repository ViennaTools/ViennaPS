#include <lsAdvect.hpp>
#include <lsToDiskMesh.hpp>

#include <geometries/psMakeTrench.hpp>
#include <psDomain.hpp>
#include <psProcess.hpp>

#include <rayTrace.hpp>

using namespace viennaps;

int main() {
  omp_set_num_threads(16);
  using NumericType = float;
  constexpr int D = 3;

  const NumericType sticking = 1.f;
  const NumericType gridDelta = .1;
  std::ofstream file("CPU_Benchmark_single.txt");
  file << "Sticking;Meshing;Tracing;Postprocessing\n";
  file << sticking << ";";

  auto domain = SmartPointer<Domain<NumericType, D>>::New();
  MakeTrench<NumericType, D>(domain, gridDelta, 10, 5, 5, 5, 0.05, 0.5, false,
                             true, Material::Si)
      .apply();

  auto mesh = SmartPointer<viennals::Mesh<NumericType>>::New();
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

  Timer timer;
  timer.start();
  mesher.apply();
  timer.finish();
  file << timer.currentDuration << ";";
  std::cout << "Meshing time: " << timer.currentDuration * 1e-6 << " ms"
            << std::endl;

  const auto &materialIds = *mesh->getCellData().getScalarData("MaterialIds");
  const auto &normals = *mesh->getCellData().getVectorData("Normals");
  tracer.setGeometry(mesh->nodes, normals, gridDelta);
  tracer.setMaterialIds(materialIds);

  timer.start();
  tracer.apply();
  timer.finish();
  file << timer.currentDuration << ";";
  std::cout << "Trace time: " << timer.currentDuration * 1e-6 << " ms"
            << std::endl;

  timer.start();
  auto &flux = tracer.getLocalData().getVectorData("flux");
  tracer.normalizeFlux(flux);
  // tracer.smoothFlux(flux);
  timer.finish();
  file << timer.currentDuration << "\n";
  std::cout << "Postprocessing time: " << timer.currentDuration * 1e-6 << " ms"
            << std::endl;

  mesh->getCellData().insertNextScalarData(flux, "flux");
  viennals::VTKWriter<NumericType>(mesh, "CPU_SingleParticleFlux_disk.vtp")
      .apply();

  file.close();
}