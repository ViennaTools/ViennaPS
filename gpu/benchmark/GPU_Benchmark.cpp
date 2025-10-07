#include <lsAdvect.hpp>
#include <lsToDiskMesh.hpp>

#include <process/psProcess.hpp>

#include <psgCreateSurfaceMesh.hpp>
#include <psgElementToPointData.hpp>
#include <raygMesh.hpp>
#include <raygTraceDisk.hpp>
#include <raygTraceLine.hpp>
#include <raygTraceTriangle.hpp>
#include <vcContext.hpp>

#include "BenchmarkGeometry.hpp"

using namespace viennaps;
using TranslatorType = std::unordered_map<unsigned long, unsigned long>;

int main() {
  omp_set_num_threads(16);
  using NumericType = float;
  constexpr int D = DIM;

  // constexpr std::array<NumericType, 10> sticking = {
  // 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.f};
  const std::array<NumericType, 1> sticking = {1.f};
  const int processSteps = 10;
  const int raysPerPoint = 1000;

  auto particle = viennaray::gpu::Particle<NumericType>();
  particle.name = "SingleParticle";
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
    file << "Sticking;Meshing;Tracing;Postprocessing;Advection;NumRays\n";

    viennaray::gpu::TraceTriangle<NumericType, D> tracer(context);
    tracer.setNumberOfRaysPerPoint(raysPerPoint);
    tracer.setUseRandomSeeds(false);
    tracer.setPipeline("GeneralPipeline", context->modulePath);
    tracer.setCallables("CallableWrapper", context->modulePath);
    tracer.setParticleCallableMap({pMap, cMap});
    tracer.insertNextParticle(particle);
    tracer.prepareParticlePrograms();

    for (auto i : sticking) {
      auto domain = MAKE_GEO<NumericType>();

      auto diskMesh = viennals::Mesh<NumericType>::New();
      viennals::ToDiskMesh<NumericType, D> diskMesher(diskMesh);

      auto elementKdTree =
          SmartPointer<KDTree<NumericType, Vec3D<NumericType>>>::New();
      auto surfMesh = viennals::Mesh<float>::New();
      gpu::CreateSurfaceMesh<NumericType, float, D> surfMesher(
          domain->getLevelSets().back(), surfMesh, elementKdTree);

      viennals::Advect<NumericType, D> advectionKernel;

      auto velocityField =
          SmartPointer<DefaultVelocityField<NumericType, D>>::New(2);
      auto translationField =
          SmartPointer<TranslationField<NumericType, D>>::New(
              velocityField, domain->getMaterialMap());
      advectionKernel.setVelocityField(translationField);

      for (const auto &ls : domain->getLevelSets()) {
        diskMesher.insertNextLevelSet(ls);
        advectionKernel.insertNextLevelSet(ls);
      }

      auto &particles = tracer.getParticles();
      particles[0].sticking = i;

      for (int j = 0; j < processSteps; j++) {
        file << i << ";";

        Timer timer;
        timer.start();
        diskMesher.apply();
        surfMesher.apply();
        auto mesh = gpu::CreateTriangleMesh(GRID_DELTA, surfMesh);
        translationField->buildKdTree(diskMesh->nodes);
        timer.finish();
        file << timer.currentDuration << ";";

        tracer.setGeometry(mesh);

        timer.start();
        tracer.apply();
        timer.finish();
        file << timer.currentDuration << ";";

        timer.start();
        auto pointData = viennals::PointData<NumericType>::New();
        gpu::ElementToPointData<NumericType, float>(
            tracer.getResults(), pointData, tracer.getParticles(),
            elementKdTree, diskMesh, surfMesh, GRID_DELTA)
            .apply();
        auto velocities = SmartPointer<std::vector<NumericType>>::New(
            std::move(*pointData->getScalarData("flux")));
        velocityField->prepare(domain, velocities, 0.);
        timer.finish();
        file << timer.currentDuration << ";";

        timer.start();
        advectionKernel.apply();
        timer.finish();
        file << timer.currentDuration << "\n";

        file << tracer.getNumberOfRays() << "\n";

        if (j == processSteps - 1) {
          auto mesh_surface =
              viennals::SmartPointer<viennals::Mesh<NumericType>>::New();
          viennals::ToSurfaceMesh<NumericType, D>(domain->getLevelSets().back(),
                                                  mesh_surface)
              .apply();
          viennals::VTKWriter<NumericType>(
              mesh_surface, "GPU_Benchmark_final_surface_triangle.vtp")
              .apply();
        }
      }
    }
    file.close();
  }

  { // Disk
    std::ofstream file("GPU_Benchmark_Disk.txt");
    file << "Sticking;Meshing;Tracing;Postprocessing;Advection;NumRays\n";

    viennaray::gpu::TraceDisk<NumericType, D> tracer(context);
    tracer.setNumberOfRaysPerPoint(raysPerPoint);
    tracer.setUseRandomSeeds(false);
    tracer.setPipeline("GeneralPipeline", context->modulePath);
    tracer.setCallables("CallableWrapper", context->modulePath);
    tracer.setParticleCallableMap({pMap, cMap});
    tracer.insertNextParticle(particle);
    tracer.prepareParticlePrograms();

    for (auto i : sticking) {
      auto domain = MAKE_GEO<NumericType>();

      auto diskMesh = viennals::Mesh<NumericType>::New();
      auto translator = SmartPointer<TranslatorType>::New();
      viennals::ToDiskMesh<NumericType, D> diskMesher(diskMesh);
      diskMesher.setTranslator(translator);

      viennals::Advect<NumericType, D> advectionKernel;

      auto velocityField =
          SmartPointer<DefaultVelocityField<NumericType, D>>::New(1);
      auto translationField =
          SmartPointer<TranslationField<NumericType, D>>::New(
              velocityField, domain->getMaterialMap());
      translationField->setTranslator(translator);
      advectionKernel.setVelocityField(translationField);

      for (const auto &ls : domain->getLevelSets()) {
        diskMesher.insertNextLevelSet(ls);
        advectionKernel.insertNextLevelSet(ls);
      }

      auto &particles = tracer.getParticles();
      particles[0].sticking = i;

      auto convertToFloat = [](std::vector<Vec3D<NumericType>> &input) {
        std::vector<Vec3Df> output;
        output.reserve(input.size());
        for (const auto &vec : input) {
          Vec3Df temp = {static_cast<float>(vec[0]), static_cast<float>(vec[1]),
                         static_cast<float>(vec[2])};
          output.emplace_back(temp);
        }
        return output;
      };

      for (int j = 0; j < processSteps; j++) {
        advectionKernel.prepareLS();
        file << i << ";";

        Timer timer;
        timer.start();
        diskMesher.apply();
        std::vector<Vec3Df> fPoints = convertToFloat(diskMesh->getNodes());
        std::vector<Vec3Df> fNormals =
            convertToFloat(*diskMesh->getCellData().getVectorData("Normals"));
        Vec3Df fMinExtent = {static_cast<float>(diskMesh->minimumExtent[0]),
                             static_cast<float>(diskMesh->minimumExtent[1]),
                             static_cast<float>(diskMesh->minimumExtent[2])};
        Vec3Df fMaxExtent = {static_cast<float>(diskMesh->maximumExtent[0]),
                             static_cast<float>(diskMesh->maximumExtent[1]),
                             static_cast<float>(diskMesh->maximumExtent[2])};
        const viennaray::gpu::DiskMesh mesh{
            .points = fPoints,
            .normals = fNormals,
            .minimumExtent = fMinExtent,
            .maximumExtent = fMaxExtent,
            .radius = static_cast<float>(domain->getGridDelta() *
                                         rayInternal::DiskFactor<D>),
            .gridDelta = static_cast<float>(domain->getGridDelta())};
        timer.finish();
        file << timer.currentDuration << ";";

        tracer.setGeometry(mesh);

        timer.start();
        tracer.apply();
        timer.finish();
        file << timer.currentDuration << ";";

        timer.start();
        int smoothingNeighbors = 1;
        std::vector<float> flux(mesh.points.size(), 0.);
        tracer.getFlux(flux.data(), 0, 0, smoothingNeighbors);
        std::vector<NumericType> fluxCasted(flux.begin(), flux.end());
        auto velocities =
            SmartPointer<std::vector<NumericType>>::New(std::move(fluxCasted));
        velocityField->prepare(domain, velocities, 0.);
        timer.finish();
        file << timer.currentDuration << ";";

        timer.start();
        advectionKernel.apply();
        timer.finish();
        file << timer.currentDuration << "\n";

        file << tracer.getNumberOfRays() << "\n";

        if (j == processSteps - 1) {
          auto mesh_surface =
              viennals::SmartPointer<viennals::Mesh<NumericType>>::New();
          viennals::ToSurfaceMesh<NumericType, D>(domain->getLevelSets().back(),
                                                  mesh_surface)
              .apply();
          viennals::VTKWriter<NumericType>(
              mesh_surface, "GPU_Benchmark_final_surface_disk.vtp")
              .apply();
        }
      }
    }
    file.close();
  }

  if (D == 2) { // Line
    std::ofstream file("GPU_Benchmark_Line.txt");
    file << "Sticking;Meshing;Tracing;Postprocessing;Advection;NumRays\n";

    viennaray::gpu::TraceLine<NumericType, D> tracer(context);
    tracer.setNumberOfRaysPerPoint(raysPerPoint);
    tracer.setUseRandomSeeds(false);
    tracer.setPipeline("GeneralPipeline", context->modulePath);
    tracer.setCallables("CallableWrapper", context->modulePath);
    tracer.setParticleCallableMap({pMap, cMap});
    tracer.insertNextParticle(particle);
    tracer.prepareParticlePrograms();

    for (auto i : sticking) {
      auto domain = MAKE_GEO<NumericType>();

      auto diskMesh = viennals::Mesh<NumericType>::New();
      viennals::ToDiskMesh<NumericType, D> diskMesher(diskMesh);

      auto elementKdTree =
          SmartPointer<KDTree<NumericType, Vec3D<NumericType>>>::New();
      auto surfMesh = viennals::Mesh<NumericType>::New();
      viennals::ToSurfaceMesh<NumericType, D> surfMesher(
          domain->getLevelSets().back(), surfMesh);

      viennals::Advect<NumericType, D> advectionKernel;

      auto velocityField =
          SmartPointer<DefaultVelocityField<NumericType, D>>::New(2);
      auto translationField =
          SmartPointer<TranslationField<NumericType, D>>::New(
              velocityField, domain->getMaterialMap());
      advectionKernel.setVelocityField(translationField);

      for (const auto &ls : domain->getLevelSets()) {
        diskMesher.insertNextLevelSet(ls);
        advectionKernel.insertNextLevelSet(ls);
      }

      auto &particles = tracer.getParticles();
      particles[0].sticking = i;

      for (int j = 0; j < processSteps; j++) {
        advectionKernel.prepareLS();
        file << i << ";";

        Timer timer;
        timer.start();
        diskMesher.apply();
        surfMesher.apply();
        std::vector<Vec3D<NumericType>> elementCenters(surfMesh->lines.size());
        for (int i = 0; i < surfMesh->lines.size(); ++i) {
          Vec3D<NumericType> p0 = {surfMesh->nodes[surfMesh->lines[i][0]][0],
                                   surfMesh->nodes[surfMesh->lines[i][0]][1],
                                   surfMesh->nodes[surfMesh->lines[i][0]][2]};
          Vec3D<NumericType> p1 = {surfMesh->nodes[surfMesh->lines[i][1]][0],
                                   surfMesh->nodes[surfMesh->lines[i][1]][1],
                                   surfMesh->nodes[surfMesh->lines[i][1]][2]};
          elementCenters[i] = (p0 + p1) / NumericType(2);
        }
        elementKdTree->setPoints(elementCenters);
        elementKdTree->build();
        std::vector<Vec3Df> fNodes(surfMesh->nodes.size());
        for (size_t i = 0; i < surfMesh->nodes.size(); i++) {
          fNodes[i] = {static_cast<float>(surfMesh->nodes[i][0]),
                       static_cast<float>(surfMesh->nodes[i][1]),
                       static_cast<float>(surfMesh->nodes[i][2])};
        }
        Vec3Df fMinExtent = {static_cast<float>(diskMesh->minimumExtent[0]),
                             static_cast<float>(diskMesh->minimumExtent[1]),
                             static_cast<float>(diskMesh->minimumExtent[2])};
        Vec3Df fMaxExtent = {static_cast<float>(diskMesh->maximumExtent[0]),
                             static_cast<float>(diskMesh->maximumExtent[1]),
                             static_cast<float>(diskMesh->maximumExtent[2])};
        float gridDelta = static_cast<float>(domain->getGridDelta());
        const viennaray::gpu::LineMesh mesh{fNodes, surfMesh->lines, fMinExtent,
                                            fMaxExtent, gridDelta};
        translationField->buildKdTree(diskMesh->nodes);
        timer.finish();
        file << timer.currentDuration << ";";

        tracer.setGeometry(mesh);

        timer.start();
        tracer.apply();
        timer.finish();
        file << timer.currentDuration << ";";

        timer.start();
        int smoothingNeighbors = 1;
        std::vector<float> elementFlux(mesh.lines.size(), 0.);
        std::vector<float> diskFlux(diskMesh->nodes.size(), 0.f);
        tracer.getFlux(elementFlux.data(), 0, 0, smoothingNeighbors);
        for (int i = 0; i < diskMesh->nodes.size(); i++) {
          auto closestPoint = elementKdTree->findNearest(diskMesh->nodes[i]);
          diskFlux[i] = elementFlux[closestPoint->first];
        }
        std::vector<NumericType> diskFluxCasted(diskFlux.begin(),
                                                diskFlux.end());
        auto velocities = SmartPointer<std::vector<NumericType>>::New(
            std::move(diskFluxCasted));
        velocityField->prepare(domain, velocities, 0.);
        timer.finish();
        file << timer.currentDuration << ";";

        timer.start();
        advectionKernel.apply();
        timer.finish();
        file << timer.currentDuration << "\n";

        file << tracer.getNumberOfRays() << "\n";

        if (j == processSteps - 1) {
          auto mesh_surface =
              viennals::SmartPointer<viennals::Mesh<NumericType>>::New();
          viennals::ToSurfaceMesh<NumericType, D>(domain->getLevelSets().back(),
                                                  mesh_surface)
              .apply();
          viennals::VTKWriter<NumericType>(
              mesh_surface, "GPU_Benchmark_final_surface_line.vtp")
              .apply();
        }
      }
    }
    file.close();
  }
}