#include <lsToDiskMesh.hpp>
#include <process/psTranslationField.hpp>
#include <psElementToPointData.hpp>
#include <rayTraceDisk.hpp>
#include <rayTraceTriangle.hpp>

#include "Benchmark.hpp"

int main() {
  using NumericType = float;
  constexpr int D = DIM;

  const std::vector<int> numThreadsList = {1, 2, 4, 8, 16};
  std::string fluxLabel = particleType == 0 ? "flux" : "ionFlux";
  auto particle = makeCPUParticle<NumericType, D>();

  if constexpr (runDisk) { // Disk
    std::ofstream file(std::string("CPU_Scaling_Disk_") +
                       std::to_string(particleType) + ".txt");
    file << "Meshing;Tracing;Postprocessing;RayTraced;NumThreads\n";

    viennaray::TraceDisk<NumericType, D> tracer;
    tracer.setNumberOfRaysPerPoint(raysPerPoint);
    if (FIXED_RAYS)
      tracer.setNumberOfRaysFixed(numRays);
    tracer.setUseRandomSeeds(false);
    tracer.setParticleType(particle);

    std::cout << "Starting Disk Benchmark\n";

    for (int i = 0; i < numThreadsList.size(); i++) {
      std::cout << "  T: " << numThreadsList[i] << "\n";
      omp_set_num_threads(numThreadsList[i]);
      auto domain = MAKE_GEO<NumericType>();

      auto diskMesh = viennals::Mesh<NumericType>::New();
      auto translator = SmartPointer<TranslatorType>::New();
      viennals::ToDiskMesh<NumericType, D> mesher(domain->getSurface(),
                                                  diskMesh);
      mesher.setTranslator(translator);

      for (int j = 0; j < numRuns; j++) {
        std::cout << "    Process Step: " << j + 1 << "\n";

        Timer timer;

        // MESHING
        timer.start();
        mesher.apply();
        const auto &materialIds = *diskMesh->getMaterialIds();
        const auto &normals = *diskMesh->getNormals();
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
        auto flux = std::move(*tracer.getLocalData().getScalarData(fluxLabel));
        tracer.normalizeFlux(flux);
        int smoothingNeighbors = 1;
        tracer.smoothFlux(flux, smoothingNeighbors);
        timer.finish();
        file << timer.currentDuration << ";";
        file << tracer.getRayTraceInfo().totalRaysTraced << ";";
        file << numThreadsList[i] << "\n";
      }
    }

    file.close();
  }

  if constexpr (runTriangle) { // Triangle
    std::ofstream file(std::string("CPU_Scaling_Triangle_") +
                       std::to_string(particleType) + ".txt");
    file << "Meshing;Tracing;Postprocessing;RayTraced;NumThreads\n";

    std::cout << "Starting Triangle Benchmark\n";

    viennaray::TraceTriangle<NumericType, D> tracer;
    tracer.setNumberOfRaysPerPoint(raysPerPoint);
    if (FIXED_RAYS)
      tracer.setNumberOfRaysFixed(numRays);
    tracer.setUseRandomSeeds(false);
    tracer.setParticleType(particle);

    const auto &dataLabels = particle->getLocalDataLabels();

    for (int i = 0; i < numThreadsList.size(); i++) {
      std::cout << "  T: " << numThreadsList[i] << "\n";
      omp_set_num_threads(numThreadsList[i]);
      auto domain = MAKE_GEO<NumericType>();

      auto diskMesh = viennals::Mesh<NumericType>::New();
      auto translator = SmartPointer<TranslatorType>::New();
      viennals::ToDiskMesh<NumericType, D> diskMesher(domain->getSurface(),
                                                      diskMesh);
      diskMesher.setTranslator(translator);

      auto elementKdTree =
          SmartPointer<KDTree<NumericType, Vec3D<NumericType>>>::New();
      auto surfMesh = viennals::Mesh<NumericType>::New();

      auto velocityField =
          SmartPointer<DefaultVelocityField<NumericType, D>>::New();
      auto translationField =
          SmartPointer<TranslationField<NumericType, D>>::New(
              velocityField, domain->getMaterialMap(), 1);
      translationField->setTranslator(translator);

      for (int j = 0; j < numRuns; j++) {
        std::cout << "    Process Step: " << j + 1 << "\n";

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
        auto pointData = PointData<NumericType>::New();
        auto fluxResult =
            std::move(*tracer.getLocalData().getScalarData(fluxLabel));
        tracer.normalizeFlux(fluxResult);
        std::vector<std::vector<NumericType>> fluxResultVec;
        fluxResultVec.push_back(std::move(fluxResult));
        if constexpr (particleType == 1) {
          fluxResult = std::move(*tracer.getLocalData().getScalarData(1));
          tracer.normalizeFlux(fluxResult);
          fluxResultVec.push_back(std::move(fluxResult));
        }
        ElementToPointData<NumericType, float, float> post(
            dataLabels, pointData, elementKdTree, diskMesh, surfMesh,
            domain->getGridDelta() * 2.0f);
        post.setElementDataArrays(std::move(fluxResultVec));
        post.apply();
        timer.finish();
        file << timer.currentDuration << ";";
        file << tracer.getRayTraceInfo().totalRaysTraced << ";";
        file << numThreadsList[i] << "\n";
      }
    }
    file.close();
  }
}