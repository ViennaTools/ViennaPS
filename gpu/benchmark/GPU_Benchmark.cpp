#include <gpu/raygTraceDisk.hpp>
#include <gpu/raygTraceTriangle.hpp>
#include <lsToDiskMesh.hpp>
#include <process/psTranslationField.hpp>
#include <psElementToPointData.hpp>

#include "Benchmark.hpp"

int main(int argc, char **argv) {
  omp_set_num_threads(16);
  using NumericType = float;
  constexpr int D = DIM;
  auto context = DeviceContext::createContext();

  bool preparePost = true;
  bool fixedRays = false;
  int particleType = 0;

  auto args = parseArgs(argc, argv);
  particleType = std::get<0>(args);
  fixedRays = std::get<1>(args);
  preparePost = std::get<2>(args);

  auto filePostFix = [&]() {
    std::string suffix = "";
    suffix += fixedRays ? "_fixedNumRays" : "_raysPerPoint";
    suffix += preparePost ? "_preparePost" : "_noPreparePost";
    suffix += particleType == 0 ? "_Neutral" : "_Ion";
    return suffix;
  };

  CudaBuffer deviceParamsBuffer;
  if (particleType == 1) {
    auto deviceParams = getDeviceParams(particleType);
    deviceParamsBuffer.allocUploadSingle(deviceParams);
  }

  if constexpr (runTriangle) { // Triangle
    std::ofstream file("GPU_Benchmark_Triangle" + filePostFix() + ".txt");
    file << "Meshing;Tracing;Postprocessing;GridDelta\n";

    viennaray::gpu::TraceTriangle<NumericType, D> tracer(context);
    tracer.setNumberOfRaysPerPoint(raysPerPoint);
    if (fixedRays)
      tracer.setNumberOfRaysFixed(numRays);
    tracer.setUseRandomSeeds(false);
    tracer.setCallables("ViennaPSCallableWrapper", context->modulePath);
    auto particleConfig = makeGPUParticle<NumericType, D>(particleType);
    tracer.insertNextParticle(std::get<0>(particleConfig));
    tracer.setParticleCallableMap(
        {std::get<1>(particleConfig), std::get<2>(particleConfig)});
    if (particleType == 1) {
      tracer.setParameters(deviceParamsBuffer.dPointer());
    }
    tracer.prepareParticlePrograms();

    const auto &dataLabels = std::get<0>(particleConfig).dataLabels;

    std::cout << "Starting Triangle Benchmark\n";

    for (auto gd : gridDeltaValues) {
      std::cout << "  Grid Delta: " << gd << "\n";
      auto domain = MAKE_GEO<NumericType>(gd);

      auto diskMesh = viennals::Mesh<NumericType>::New();
      auto translator = SmartPointer<TranslatorType>::New();
      viennals::ToDiskMesh<NumericType, D> diskMesher(domain->getSurface(),
                                                      diskMesh);
      diskMesher.setTranslator(translator);

      auto elementKdTree =
          SmartPointer<KDTree<NumericType, Vec3D<NumericType>>>::New();
      auto surfMesh = viennals::Mesh<float>::New();

      viennals::Advect<NumericType, D> advectionKernel;

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
        auto pointData = PointData<NumericType>::New();
        ElementToPointData<NumericType, float, viennaray::gpu::ResultType> post(
            dataLabels, pointData, elementKdTree, diskMesh, surfMesh,
            domain->getGridDelta() * 2.0f);
        if (preparePost)
          post.prepare();
        tracer.syncStreams();
        timer.finish();
        file << timer.currentDuration << ";";

        // POSTPROCESSING
        timer.start();
        tracer.normalizeResults();
        post.setElementDataArrays(tracer.getResults());
        if (!preparePost)
          post.prepare();
        post.convert();
        timer.finish();
        file << timer.currentDuration << ";";

        file << domain->getGridDelta() << "\n";
      }
    }
    file.close();
  }

  if constexpr (runDisk) { // Disk
    std::ofstream file("GPU_Benchmark_Disk" + filePostFix() + ".txt");
    file << "Meshing;Tracing;Postprocessing;GridDelta\n";

    viennaray::gpu::TraceDisk<NumericType, D> tracer(context);
    tracer.setNumberOfRaysPerPoint(raysPerPoint);
    if (fixedRays)
      tracer.setNumberOfRaysFixed(numRays);
    tracer.setUseRandomSeeds(false);
    tracer.setCallables("ViennaPSCallableWrapper", context->modulePath);
    auto particleConfig = makeGPUParticle<NumericType, D>(particleType);
    tracer.insertNextParticle(std::get<0>(particleConfig));
    tracer.setParticleCallableMap(
        {std::get<1>(particleConfig), std::get<2>(particleConfig)});
    if (particleType == 1) {
      tracer.setParameters(deviceParamsBuffer.dPointer());
    }
    tracer.prepareParticlePrograms();

    std::cout << "Starting Disk Benchmark\n";

    for (auto gd : gridDeltaValues) {
      std::cout << "  Grid Delta: " << gd << "\n";
      auto domain = MAKE_GEO<NumericType>(gd);

      auto diskMesh = viennals::Mesh<NumericType>::New();
      auto translator = SmartPointer<TranslatorType>::New();
      viennals::ToDiskMesh<NumericType, D> diskMesher(domain->getSurface(),
                                                      diskMesh);
      diskMesher.setTranslator(translator);

      for (int j = 0; j < numRuns; j++) {
        std::cout << "    Process Step: " << j + 1 << "\n";

        Timer timer;

        // MESHING
        timer.start();
        diskMesher.apply();
        viennaray::DiskMesh mesh(diskMesh->nodes, *diskMesh->getNormals(),
                                 domain->getGridDelta());
        mesh.minimumExtent = diskMesh->minimumExtent;
        mesh.maximumExtent = diskMesh->maximumExtent;
        mesh.radius = static_cast<float>(domain->getGridDelta() *
                                         rayInternal::DiskFactor<D>);
        tracer.setGeometry(mesh);
        timer.finish();
        file << timer.currentDuration << ";";

        // TRACING
        timer.start();
        tracer.apply();
        tracer.syncStreams();
        timer.finish();
        file << timer.currentDuration << ";";

        // POSTPROCESSING
        timer.start();
        tracer.normalizeResults();
        tracer.downloadResults();
        int smoothingNeighbors = 1;
        auto flux = tracer.getFlux(0, 0, smoothingNeighbors);
        timer.finish();
        file << timer.currentDuration << ";";

        file << domain->getGridDelta() << "\n";
      }
    }
    file.close();
  }
}