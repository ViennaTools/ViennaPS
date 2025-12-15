#include <lsAdvect.hpp>
#include <lsToDiskMesh.hpp>

#include <process/psProcess.hpp>
#include <rayTraceDisk.hpp>
#include <rayTraceTriangle.hpp>

#include "Benchmark.hpp"

using namespace viennaps;

int main() {
  omp_set_num_threads(16);
  using NumericType = float;
  constexpr int D = DIM;

  // const std::array<NumericType, 4> gridDeltaValues = {0.05f, 0.1f, 0.2f,
  // 0.4f};
  auto gridDeltaValues = linspace<NumericType, 8>(0.01f, 0.4f);
  const int numRuns = 20;
  const int raysPerPoint = 1000;
  const int numRays = int(1.4e8);

  auto particle = makeCPUParticle<NumericType, D>();
  std::vector<std::unique_ptr<viennaray::AbstractParticle<NumericType>>>
      particles;
  particles.push_back(particle->clone());
  std::string fluxLabel = particleType == 0 ? "flux" : "ionFlux";

  { // Disk
    std::ofstream file("CPU_Benchmark_Disk.txt");
    file << "Meshing;Tracing;Postprocessing;GridDelta\n";

    viennaray::TraceDisk<NumericType, D> tracer;
    tracer.setNumberOfRaysPerPoint(raysPerPoint);
    if (FIXED_RAYS)
      tracer.setNumberOfRaysFixed(numRays);
    tracer.setUseRandomSeeds(false);
    tracer.setParticleType(particle);

    std::cout << "Starting Disk Benchmark\n";

    for (int i = 0; i < gridDeltaValues.size(); i++) {
      std::cout << "  Grid Delta: " << gridDeltaValues[i] << "\n";
      auto domain = MAKE_GEO<NumericType>(gridDeltaValues[i]);

      auto diskMesh = viennals::Mesh<NumericType>::New();
      auto translator = SmartPointer<TranslatorType>::New();
      viennals::ToDiskMesh<NumericType, D> mesher(diskMesh);
      mesher.setTranslator(translator);

      viennals::Advect<NumericType, D> advectionKernel;

      auto velocityField =
          SmartPointer<DefaultVelocityField<NumericType, D>>::New();
      auto translationField =
          SmartPointer<TranslationField<NumericType, D>>::New(
              velocityField, domain->getMaterialMap(), 1);
      translationField->setTranslator(translator);
      advectionKernel.setVelocityField(translationField);

      for (const auto ls : domain->getLevelSets()) {
        mesher.insertNextLevelSet(ls);
        advectionKernel.insertNextLevelSet(ls);
      }

      for (int j = 0; j < numRuns; j++) {
        std::cout << "    Process Step: " << j + 1 << "\n";
        advectionKernel.prepareLS();

        Timer timer;

        // MESHING
        timer.start();
        mesher.apply();
        const auto &materialIds =
            *diskMesh->getCellData().getScalarData("MaterialIds");
        const auto &normals = *diskMesh->getCellData().getVectorData("Normals");
        tracer.setGeometry(diskMesh->nodes, normals, domain->getGridDelta());
        tracer.setMaterialIds(materialIds);
        timer.finish();
        file << timer.currentDuration << ";";

        // TRACING
        timer.start();
        tracer.apply();
        timer.finish();
        file << timer.currentDuration << ";";

        // POSTPROCESSING
        timer.start();
        auto &flux = tracer.getLocalData().getVectorData(fluxLabel);
        tracer.normalizeFlux(flux);
        int smoothingNeighbors = 1;
        tracer.smoothFlux(flux, smoothingNeighbors);
        auto velocities =
            SmartPointer<std::vector<NumericType>>::New(std::move(flux));
        velocityField->prepare(domain, velocities, 0.);
        timer.finish();
        file << timer.currentDuration << ";";

        // // ADVECTION
        // timer.start();
        // advectionKernel.apply();
        // timer.finish();
        // file << timer.currentDuration << ";";

        file << gridDeltaValues[i] << "\n";
      }
    }

    file.close();
  }

  { // Triangle
    std::ofstream file("CPU_Benchmark_Triangle.txt");
    file << "Meshing;Tracing;Postprocessing;GridDelta\n";

    std::cout << "Starting Triangle Benchmark\n";

    viennaray::TraceTriangle<NumericType, D> tracer;
    tracer.setNumberOfRaysPerPoint(raysPerPoint);
    if (FIXED_RAYS)
      tracer.setNumberOfRaysFixed(numRays);
    tracer.setUseRandomSeeds(false);
    tracer.setParticleType(particle);

    for (int i = 0; i < gridDeltaValues.size(); i++) {
      std::cout << "  Grid Delta: " << gridDeltaValues[i] << "\n";
      auto domain = MAKE_GEO<NumericType>(gridDeltaValues[i]);

      auto diskMesh = viennals::Mesh<NumericType>::New();
      auto translator = SmartPointer<TranslatorType>::New();
      viennals::ToDiskMesh<NumericType, D> diskMesher(diskMesh);
      diskMesher.setTranslator(translator);

      auto elementKdTree =
          SmartPointer<KDTree<NumericType, Vec3D<NumericType>>>::New();
      auto surfMesh = viennals::Mesh<NumericType>::New();

      viennals::Advect<NumericType, D> advectionKernel;

      auto velocityField =
          SmartPointer<DefaultVelocityField<NumericType, D>>::New();
      auto translationField =
          SmartPointer<TranslationField<NumericType, D>>::New(
              velocityField, domain->getMaterialMap(), 1);
      translationField->setTranslator(translator);
      advectionKernel.setVelocityField(translationField);

      for (const auto ls : domain->getLevelSets()) {
        diskMesher.insertNextLevelSet(ls);
        advectionKernel.insertNextLevelSet(ls);
      }

      for (int j = 0; j < numRuns; j++) {
        std::cout << "    Process Step: " << j + 1 << "\n";
        advectionKernel.prepareLS();

        Timer timer;

        // MESHING
        timer.start();
        diskMesher.apply();
        translationField->buildKdTree(diskMesh->nodes);
        setupTriangleGeometry<NumericType, D, decltype(tracer)>(
            domain, surfMesh, elementKdTree, tracer);
        timer.finish();
        file << timer.currentDuration << ";";

        // TRACING
        timer.start();
        tracer.apply();
        timer.finish();
        file << timer.currentDuration << ";";

        // POSTPROCESSING
        timer.start();
        auto pointData = viennals::PointData<NumericType>::New();
        auto fluxResult = tracer.getLocalData().getVectorData(0);
        tracer.normalizeFlux(fluxResult);
        std::vector<std::vector<NumericType>> fluxResultVec;
        fluxResultVec.push_back(std::move(fluxResult));
        if constexpr (particleType == 1) {
          fluxResult = tracer.getLocalData().getVectorData(1);
          tracer.normalizeFlux(fluxResult);
          fluxResultVec.push_back(std::move(fluxResult));
        }
        ElementToPointData<NumericType, float, float>(
            IndexMap(particles), fluxResultVec, pointData, elementKdTree,
            diskMesh, surfMesh, domain->getGridDelta() * 2.0f)
            .apply();
        auto velocities = SmartPointer<std::vector<NumericType>>::New(
            std::move(*pointData->getScalarData(fluxLabel)));
        velocityField->prepare(domain, velocities, 0.);
        timer.finish();
        file << timer.currentDuration << ";";

        // // ADVECTION
        // timer.start();
        // advectionKernel.apply();
        // timer.finish();
        // file << timer.currentDuration << ";";

        file << gridDeltaValues[i] << "\n";
      }
    }
    file.close();
  }
}