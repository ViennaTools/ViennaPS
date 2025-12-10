#include <lsAdvect.hpp>
#include <lsToDiskMesh.hpp>

#include <process/psProcess.hpp>

#include <psCreateSurfaceMesh.hpp>
#include <psElementToPointData.hpp>
#include <rayMesh.hpp>
#include <raygTraceDisk.hpp>
#include <raygTraceLine.hpp>
#include <raygTraceTriangle.hpp>
#include <vcContext.hpp>

#include "Benchmark.hpp"

using namespace viennaps;

int main() {
  omp_set_num_threads(16);
  using NumericType = float;
  constexpr int D = DIM;

  const NumericType cSticking = DEFAULT_STICKING;
  // const std::array<NumericType, 4> gridDeltaValues = {0.05f, 0.1f, 0.2f,
  // 0.4f};
  auto gridDeltaValues = linspace<NumericType, 8>(0.01f, 0.4f);
  const int numRuns = 20;
  const int raysPerPoint = 1000;
  const int numRays = int(1.4e8);

  auto context = DeviceContext::createContext();

  CudaBuffer deviceParamsBuffer;
  if constexpr (particleType == 1) {
    auto deviceParams = getDeviceParams();
    deviceParamsBuffer.allocUploadSingle(deviceParams);
  }

  { // Triangle
    std::ofstream file("GPU_Benchmark_Triangle.txt");
    file << "Meshing;Tracing;Postprocessing;GridDelta\n";

    viennaray::gpu::TraceTriangle<NumericType, D> tracer(context);
    tracer.setNumberOfRaysPerPoint(raysPerPoint);
    if (FIXED_RAYS)
      tracer.setNumberOfRaysFixed(numRays);
    tracer.setUseRandomSeeds(false);
    tracer.setCallables("CallableWrapper", context->modulePath);
    auto particleConfig = makeGPUParticle<NumericType, D>();
    tracer.insertNextParticle(std::get<0>(particleConfig));
    tracer.setParticleCallableMap(
        {std::get<1>(particleConfig), std::get<2>(particleConfig)});
    if constexpr (particleType == 1) {
      tracer.setParameters(deviceParamsBuffer.dPointer());
    }
    tracer.prepareParticlePrograms();

    std::cout << "Starting Triangle Benchmark\n";

    for (auto gd : gridDeltaValues) {
      std::cout << "  Grid Delta: " << gd << "\n";
      auto domain = MAKE_GEO<NumericType>(gd);

      auto diskMesh = viennals::Mesh<NumericType>::New();
      auto translator = SmartPointer<TranslatorType>::New();
      viennals::ToDiskMesh<NumericType, D> diskMesher(diskMesh);
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
      advectionKernel.setVelocityField(translationField);

      for (const auto &ls : domain->getLevelSets()) {
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
        tracer.normalizeResults();
        // tracer.downloadResults();
        gpu::ElementToPointData<NumericType, float, viennaray::gpu::ResultType>(
            tracer.getResults(), pointData, tracer.getParticles(),
            elementKdTree, diskMesh, surfMesh, domain->getGridDelta() * 2.0f)
            .apply();
        auto velocities = SmartPointer<std::vector<NumericType>>::New(
            std::move(*pointData->getScalarData("flux")));
        velocityField->prepare(domain, velocities, 0.);
        timer.finish();
        file << timer.currentDuration << ";";

        // // ADVECTION
        // timer.start();
        // advectionKernel.apply();
        // timer.finish();
        // file << timer.currentDuration << ";";

        file << domain->getGridDelta() << "\n";
      }
    }
    file.close();
  }

  { // Disk
    std::ofstream file("GPU_Benchmark_Disk.txt");
    file << "Meshing;Tracing;Postprocessing;GridDelta\n";

    viennaray::gpu::TraceDisk<NumericType, D> tracer(context);
    tracer.setNumberOfRaysPerPoint(raysPerPoint);
    if (FIXED_RAYS)
      tracer.setNumberOfRaysFixed(numRays);
    tracer.setUseRandomSeeds(false);
    tracer.setCallables("CallableWrapper", context->modulePath);
    auto particleConfig = makeGPUParticle<NumericType, D>();
    tracer.insertNextParticle(std::get<0>(particleConfig));
    tracer.setParticleCallableMap(
        {std::get<1>(particleConfig), std::get<2>(particleConfig)});
    if constexpr (particleType == 1) {
      tracer.setParameters(deviceParamsBuffer.dPointer());
    }
    tracer.prepareParticlePrograms();

    std::cout << "Starting Disk Benchmark\n";

    for (auto gd : gridDeltaValues) {
      std::cout << "  Grid Delta: " << gd << "\n";
      auto domain = MAKE_GEO<NumericType>(gd);

      auto diskMesh = viennals::Mesh<NumericType>::New();
      auto translator = SmartPointer<TranslatorType>::New();
      viennals::ToDiskMesh<NumericType, D> diskMesher(diskMesh);
      diskMesher.setTranslator(translator);

      viennals::Advect<NumericType, D> advectionKernel;

      auto velocityField =
          SmartPointer<DefaultVelocityField<NumericType, D>>::New();
      auto translationField =
          SmartPointer<TranslationField<NumericType, D>>::New(
              velocityField, domain->getMaterialMap(), 1);
      translationField->setTranslator(translator);
      advectionKernel.setVelocityField(translationField);

      for (const auto &ls : domain->getLevelSets()) {
        diskMesher.insertNextLevelSet(ls);
        advectionKernel.insertNextLevelSet(ls);
      }

      for (int j = 0; j < numRuns; j++) {
        std::cout << "    Process Step: " << j + 1 << "\n";
        advectionKernel.prepareLS();
        file << cSticking << ";";

        Timer timer;

        // MESHING
        timer.start();
        diskMesher.apply();
        viennaray::DiskMesh mesh(
            diskMesh->nodes, *diskMesh->getCellData().getVectorData("Normals"),
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
        timer.finish();
        file << timer.currentDuration << ";";

        // POSTPROCESSING
        timer.start();
        tracer.normalizeResults();
        tracer.downloadResults();
        int smoothingNeighbors = 1;
        auto flux = tracer.getFlux(0, 0, smoothingNeighbors);
        std::vector<NumericType> fluxNumeric(flux.begin(), flux.end());
        auto velocities =
            SmartPointer<std::vector<NumericType>>::New(std::move(fluxNumeric));
        velocityField->prepare(domain, velocities, 0.);
        timer.finish();
        file << timer.currentDuration << ";";

        // // ADVECTION
        // timer.start();
        // advectionKernel.apply();
        // timer.finish();
        // file << timer.currentDuration << ";";

        file << domain->getGridDelta() << "\n";
      }
    }
    file.close();
  }

  if constexpr (D == 2) { // Line
    std::ofstream file("GPU_Benchmark_Line.txt");
    file << "Meshing;Tracing;Postprocessing;GridDelta\n";

    viennaray::gpu::TraceLine<NumericType, D> tracer(context);
    tracer.setNumberOfRaysPerPoint(raysPerPoint);
    // tracer.setNumberOfRaysFixed(numRays);
    tracer.setUseRandomSeeds(false);
    tracer.setCallables("CallableWrapper", context->modulePath);
    auto particleConfig = makeGPUParticle<NumericType, D>();
    tracer.insertNextParticle(std::get<0>(particleConfig));
    tracer.setParticleCallableMap(
        {std::get<1>(particleConfig), std::get<2>(particleConfig)});
    if constexpr (particleType == 1) {
      tracer.setParameters(deviceParamsBuffer.dPointer());
    }
    tracer.prepareParticlePrograms();

    std::cout << "Starting Line Benchmark\n";

    for (auto gd : gridDeltaValues) {
      std::cout << "  Grid Delta: " << gd << "\n";
      auto domain = MAKE_GEO<NumericType>(gd);

      auto diskMesh = viennals::Mesh<NumericType>::New();
      auto translator = SmartPointer<TranslatorType>::New();
      viennals::ToDiskMesh<NumericType, D> diskMesher(diskMesh);
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
      advectionKernel.setVelocityField(translationField);

      for (const auto &ls : domain->getLevelSets()) {
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
        setupLineGeometry<NumericType, D, decltype(tracer)>(
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
        postProcessLineData<NumericType, decltype(tracer)>(
            *pointData, diskMesh, 2, domain->getGridDelta(), tracer,
            elementKdTree, surfMesh);
        auto velocities = SmartPointer<std::vector<NumericType>>::New(
            std::move(*pointData->getScalarData("flux")));
        velocityField->prepare(domain, velocities, 0.);
        timer.finish();
        file << timer.currentDuration << ";";

        // // ADVECTION
        // timer.start();
        // advectionKernel.apply();
        // timer.finish();
        // file << timer.currentDuration << ";";

        file << domain->getGridDelta() << "\n";
      }
    }
    file.close();
  }
}