#include <lsAdvect.hpp>
#include <lsToDiskMesh.hpp>

#include <process/psProcess.hpp>

#include <psCreateSurfaceMesh.hpp>
#include <psElementToPointData.hpp>
#include <raygMesh.hpp>
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
  auto gridDeltaValues = linspace<NumericType, 6>(0.05f, 0.4f);
  const int numRuns = 10;
  const int raysPerPoint = 1000;
  // const int numRays = int(1.4e8);

  auto particle = viennaray::gpu::Particle<NumericType>();
  particle.name = "SingleParticle";
  particle.sticking = cSticking;
  particle.dataLabels.emplace_back("flux");
  std::unordered_map<std::string, unsigned> pMap = {{"SingleParticle", 0}};
  std::vector<viennaray::gpu::CallableConfig> cMap = {
      {0, viennaray::gpu::CallableSlot::COLLISION,
       "__direct_callable__singleNeutralCollision"},
      {0, viennaray::gpu::CallableSlot::REFLECTION,
       "__direct_callable__singleNeutralReflection"}};

  auto context = DeviceContext::createContext();

  if (D == 3) { // Triangle
    std::ofstream file("GPU_Benchmark_Triangle.txt");
    file << "Meshing;Tracing;Postprocessing;GridDelta\n";

    viennaray::gpu::TraceTriangle<NumericType, D> tracer(context);
    tracer.setNumberOfRaysPerPoint(raysPerPoint);
    // tracer.setNumberOfRaysFixed(numRays);
    tracer.setUseRandomSeeds(false);
    tracer.setCallables("CallableWrapper", context->modulePath);
    auto particleConfig = makeGPUParticle<NumericType, D>();
    tracer.insertNextParticle(std::get<0>(particleConfig));
    tracer.setParticleCallableMap(
        {std::get<1>(particleConfig), std::get<2>(particleConfig)});
    if constexpr (particleType == 1) {
      auto deviceParams = getDeviceParams();
      CudaBuffer deviceParamsBuffer;
      deviceParamsBuffer.allocUploadSingle(deviceParams);
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
      CreateSurfaceMesh<NumericType, float, D> surfMesher(
          domain->getLevelSets().back(), surfMesh, elementKdTree);

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
        surfMesher.apply();
        translationField->buildKdTree(diskMesh->nodes);
        auto mesh = gpu::CreateTriangleMesh(domain->getGridDelta(), surfMesh);
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
        auto pointData = viennals::PointData<NumericType>::New();
        gpu::ElementToPointData<NumericType, float>(
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

        // if (j == numRuns - 1) {
        //   auto mesh_surface =
        //       viennals::SmartPointer<viennals::Mesh<NumericType>>::New();
        //   viennals::ToSurfaceMesh<NumericType,
        //   D>(domain->getLevelSets().back(),
        //                                           mesh_surface)
        //       .apply();
        //   viennals::VTKWriter<NumericType>(
        //       mesh_surface, "GPU_Benchmark_final_surface_triangle.vtp")
        //       .apply();
        // }
      }
    }
    file.close();
  }

  { // Disk
    std::ofstream file("GPU_Benchmark_Disk.txt");
    file << "Meshing;Tracing;Postprocessing;GridDelta\n";

    viennaray::gpu::TraceDisk<NumericType, D> tracer(context);
    tracer.setNumberOfRaysPerPoint(raysPerPoint);
    // tracer.setNumberOfRaysFixed(numRays);
    tracer.setUseRandomSeeds(false);
    tracer.setCallables("CallableWrapper", context->modulePath);
    auto particleConfig = makeGPUParticle<NumericType, D>();
    tracer.insertNextParticle(std::get<0>(particleConfig));
    tracer.setParticleCallableMap(
        {std::get<1>(particleConfig), std::get<2>(particleConfig)});
    if constexpr (particleType == 1) {
      auto deviceParams = getDeviceParams();
      CudaBuffer deviceParamsBuffer;
      deviceParamsBuffer.allocUploadSingle(deviceParams);
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
        const viennaray::gpu::DiskMesh mesh{
            .nodes = diskMesh->nodes,
            .normals = *diskMesh->getCellData().getVectorData("Normals"),
            .minimumExtent = diskMesh->minimumExtent,
            .maximumExtent = diskMesh->maximumExtent,
            .radius = static_cast<float>(domain->getGridDelta() *
                                         rayInternal::DiskFactor<D>),
            .gridDelta = static_cast<float>(domain->getGridDelta())};
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
        int smoothingNeighbors = 1;
        std::vector<float> flux(mesh.nodes.size(), 0.);
        tracer.getFlux(flux.data(), 0, 0, smoothingNeighbors);
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

        file << domain->getGridDelta() << "\n";

        // if (j == numRuns - 1) {
        //   auto mesh_surface =
        //       viennals::SmartPointer<viennals::Mesh<NumericType>>::New();
        //   viennals::ToSurfaceMesh<NumericType,
        //   D>(domain->getLevelSets().back(),
        //                                           mesh_surface)
        //       .apply();
        //   viennals::VTKWriter<NumericType>(
        //       mesh_surface, "GPU_Benchmark_final_surface_disk.vtp")
        //       .apply();
        // }
      }
    }
    file.close();
  }

  // if (D == 2) { // Line
  //   std::ofstream file("GPU_Benchmark_Line.txt");
  //   file << "Sticking;Meshing;Tracing;Postprocessing;Advection;NumRays\n";

  //   viennaray::gpu::TraceLine<NumericType, D> tracer(context);
  //   tracer.setNumberOfRaysPerPoint(raysPerPoint);
  //   tracer.setNumberOfRaysFixed(numRays);
  //   tracer.setUseRandomSeeds(false);
  //   tracer.setCallables("CallableWrapper", context->modulePath);
  //   tracer.setParticleCallableMap({pMap, cMap});
  //   tracer.insertNextParticle(particle);
  //   tracer.prepareParticlePrograms();

  //   for (auto i : sticking) {
  //     auto domain = MAKE_GEO<NumericType>();

  //     auto diskMesh = viennals::Mesh<NumericType>::New();
  //     viennals::ToDiskMesh<NumericType, D> diskMesher(diskMesh);

  //     auto elementKdTree =
  //         SmartPointer<KDTree<NumericType, Vec3D<NumericType>>>::New();
  //     auto surfMesh = viennals::Mesh<NumericType>::New();
  //     viennals::ToSurfaceMesh<NumericType, D> surfMesher(
  //         domain->getLevelSets().back(), surfMesh);

  //     viennals::Advect<NumericType, D> advectionKernel;

  //     auto velocityField =
  //         SmartPointer<DefaultVelocityField<NumericType, D>>::New();
  //     auto translationField =
  //         SmartPointer<TranslationField<NumericType, D>>::New(
  //             velocityField, domain->getMaterialMap(), 1);
  //     advectionKernel.setVelocityField(translationField);

  //     for (const auto &ls : domain->getLevelSets()) {
  //       diskMesher.insertNextLevelSet(ls);
  //       advectionKernel.insertNextLevelSet(ls);
  //     }

  //     auto &particles = tracer.getParticles();
  //     particles[0].sticking = i;

  //     for (int j = 0; j < numRuns; j++) {
  //       advectionKernel.prepareLS();
  //       file << i << ";";

  //       Timer timer;
  //       timer.start();
  //       diskMesher.apply();
  //       surfMesher.apply();
  //       std::vector<Vec3D<NumericType>>
  //       elementCenters(surfMesh->lines.size()); for (int i = 0; i <
  //       surfMesh->lines.size(); ++i) {
  //         Vec3D<NumericType> p0 = {surfMesh->nodes[surfMesh->lines[i][0]][0],
  //                                  surfMesh->nodes[surfMesh->lines[i][0]][1],
  //                                  surfMesh->nodes[surfMesh->lines[i][0]][2]};
  //         Vec3D<NumericType> p1 = {surfMesh->nodes[surfMesh->lines[i][1]][0],
  //                                  surfMesh->nodes[surfMesh->lines[i][1]][1],
  //                                  surfMesh->nodes[surfMesh->lines[i][1]][2]};
  //         elementCenters[i] = (p0 + p1) / NumericType(2);
  //       }
  //       elementKdTree->setPoints(elementCenters);
  //       elementKdTree->build();
  //       std::vector<Vec3Df> fNodes(surfMesh->nodes.size());
  //       for (size_t i = 0; i < surfMesh->nodes.size(); i++) {
  //         fNodes[i] = {static_cast<float>(surfMesh->nodes[i][0]),
  //                      static_cast<float>(surfMesh->nodes[i][1]),
  //                      static_cast<float>(surfMesh->nodes[i][2])};
  //       }
  //       Vec3Df fMinExtent = {static_cast<float>(diskMesh->minimumExtent[0]),
  //                            static_cast<float>(diskMesh->minimumExtent[1]),
  //                            static_cast<float>(diskMesh->minimumExtent[2])};
  //       Vec3Df fMaxExtent = {static_cast<float>(diskMesh->maximumExtent[0]),
  //                            static_cast<float>(diskMesh->maximumExtent[1]),
  //                            static_cast<float>(diskMesh->maximumExtent[2])};
  //       float gridDelta = static_cast<float>(domain->getGridDelta());
  //       const viennaray::gpu::LineMesh mesh{fNodes, surfMesh->lines,
  //       fMinExtent,
  //                                           fMaxExtent, gridDelta};
  //       translationField->buildKdTree(diskMesh->nodes);
  //       timer.finish();
  //       file << timer.currentDuration << ";";

  //       tracer.setGeometry(mesh);

  //       timer.start();
  //       tracer.apply();
  //       timer.finish();
  //       file << timer.currentDuration << ";";

  //       timer.start();
  //       int smoothingNeighbors = 1;
  //       std::vector<float> elementFlux(mesh.lines.size(), 0.);
  //       std::vector<float> diskFlux(diskMesh->nodes.size(), 0.f);
  //       tracer.getFlux(elementFlux.data(), 0, 0, smoothingNeighbors);
  //       for (int i = 0; i < diskMesh->nodes.size(); i++) {
  //         auto closestPoint = elementKdTree->findNearest(diskMesh->nodes[i]);
  //         diskFlux[i] = elementFlux[closestPoint->first];
  //       }
  //       std::vector<NumericType> diskFluxCasted(diskFlux.begin(),
  //                                               diskFlux.end());
  //       auto velocities = SmartPointer<std::vector<NumericType>>::New(
  //           std::move(diskFluxCasted));
  //       velocityField->prepare(domain, velocities, 0.);
  //       timer.finish();
  //       file << timer.currentDuration << ";";

  //       timer.start();
  //       advectionKernel.apply();
  //       timer.finish();
  //       file << timer.currentDuration;

  //       file << tracer.getNumberOfRays() << "\n";

  //       if (j == numRuns - 1) {
  //         auto mesh_surface =
  //             viennals::SmartPointer<viennals::Mesh<NumericType>>::New();
  //         viennals::ToSurfaceMesh<NumericType,
  //         D>(domain->getLevelSets().back(),
  //                                                 mesh_surface)
  //             .apply();
  //         viennals::VTKWriter<NumericType>(
  //             mesh_surface, "GPU_Benchmark_final_surface_line.vtp")
  //             .apply();
  //       }
  //     }
  //   }
  //   file.close();
  // }
}