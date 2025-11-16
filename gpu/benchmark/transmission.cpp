#include <psCreateSurfaceMesh.hpp>
#include <psDomain.hpp>
#include <rayTraceTriangle.hpp>
#include <raygTraceTriangle.hpp>

constexpr int D = 3;
using NumericType = double;

auto createCylinderMesh(NumericType gridDelta, NumericType radius,
                        NumericType length) {
  using namespace viennaps;

  double bounds[2 * D] = {-1.0, 1.0, -1.0, 1.0, 0.0, length + gridDelta};
  BoundaryType boundaryConditions[D] = {BoundaryType::INFINITE_BOUNDARY,
                                        BoundaryType::INFINITE_BOUNDARY,
                                        BoundaryType::REFLECTIVE_BOUNDARY};
  auto cylinder = viennals::Domain<NumericType, D>::New(
      bounds, boundaryConditions, gridDelta);
  viennals::MakeGeometry<NumericType, D>(
      cylinder, viennals::Cylinder<NumericType, D>::New(
                    std::array<NumericType, D>{0.0, 0.0, 0.0},
                    std::array<NumericType, D>{0.0, 0.0, 1.0},
                    length + gridDelta * 2, radius))
      .apply();
  viennals::BooleanOperation<NumericType, D>(
      cylinder, viennals::BooleanOperationEnum::INVERT)
      .apply();

  auto lsMesh = viennals::Mesh<float>::New();
  viennaps::CreateSurfaceMesh<NumericType, float, D>(cylinder, lsMesh).apply();
  std::vector<float> materialIds(lsMesh->triangles.size(), 0);

  // add plane on bottom (z=0) with material id 1
  auto normals = lsMesh->getCellData().getVectorData("Normals");
  constexpr std::array<Vec3D<float>, 4> planeNodes = {
      Vec3D<float>{-1.0f, -1.0f, 0.0f}, Vec3D<float>{1.0f, -1.0f, 0.0f},
      Vec3D<float>{1.0f, 1.0f, 0.0f}, Vec3D<float>{-1.0f, 1.0f, 0.0f}};
  constexpr std::array<std::array<unsigned, 3>, 2> planeTriangles = {
      std::array<unsigned, 3>{0, 1, 2},
      std::array<unsigned, 3>{0, 2, 3},
  };
  std::array<unsigned, 4> planeNodeIds;
  for (unsigned i = 0; i < 4; ++i) {
    auto node = planeNodes[i] * static_cast<float>(radius);
    planeNodeIds[i] = lsMesh->insertNextNode(node);
  }
  for (unsigned i = 0; i < 2; ++i) {
    auto triangle = planeTriangles[i];
    for (unsigned j = 0; j < 3; ++j) {
      triangle[j] = planeNodeIds[triangle[j]];
    }
    lsMesh->insertNextTriangle(triangle);
    materialIds.push_back(1);
    normals->push_back(Vec3D<float>{0.0f, 0.0f, 1.0f});
  }

  // add top plane (z=length) with material id 2
  for (unsigned i = 0; i < 4; ++i) {
    auto node = planeNodes[i] * static_cast<float>(radius);
    node[2] = static_cast<float>(length);
    planeNodeIds[i] = lsMesh->insertNextNode(node);
  }
  for (unsigned i = 0; i < 2; ++i) {
    auto triangle = planeTriangles[i];
    for (unsigned j = 0; j < 3; ++j) {
      triangle[j] = planeNodeIds[triangle[j]];
    }
    lsMesh->insertNextTriangle(triangle);
    materialIds.push_back(2);
    normals->push_back(Vec3D<float>{0.0f, 0.0f, -1.0f});
  }
  lsMesh->getCellData().insertNextScalarData(materialIds, "MaterialIDs");

  viennals::VTKWriter<float>(lsMesh, "cylinder.vtp").apply();

  auto mesh = viennaps::CreateTriangleMesh(gridDelta, lsMesh);
  return std::make_pair(mesh, materialIds);
}

class TransmissionParticle
    : public viennaray::Particle<TransmissionParticle, NumericType> {
public:
  std::pair<NumericType, viennaray::Vec3D<NumericType>>
  surfaceReflection(NumericType rayWeight,
                    const viennaray::Vec3D<NumericType> &rayDir,
                    const viennaray::Vec3D<NumericType> &geomNormal,
                    const unsigned int primId, const int materialId,
                    const viennaray::TracingData<NumericType> *globalData,
                    viennaray::RNG &rngState) override {
    using namespace viennaray;
    if (materialId == 1 || materialId == 2) {
      // Bottom (transmission) or Top (reflection)
      return VIENNARAY_PARTICLE_STOP;
    }
    auto direction =
        viennaray::ReflectionDiffuse<NumericType, D>(geomNormal, rngState);
    return {0.0, direction};
  }
  void surfaceCollision(
      NumericType rayWeight, const viennaray::Vec3D<NumericType> &rayDir,
      const viennaray::Vec3D<NumericType> &geomNormal,
      const unsigned int primID, const int materialId,
      viennaray::TracingData<NumericType> &localData,
      const viennaray::TracingData<NumericType> *globalData,
      viennaray::RNG &rngState) override { // collect data for this hit
    if (materialId == 1) {
      // Bottom (transmission)
      localData.getVectorData(0)[0] += 1.0;
    } else if (materialId == 2) {
      // Top (reflection)
      localData.getVectorData(0)[1] += 1.0;
    }
  }
  std::vector<std::string> getLocalDataLabels() const override {
    return {"Counts"};
  }
};

class TransmissionSource : public viennaray::Source<NumericType> {
public:
  TransmissionSource(NumericType radius, NumericType height)
      : radius_(radius), height_(height) {}

  std::array<viennaray::Vec3D<NumericType>, 2>
  getOriginAndDirection(size_t idx, viennaray::RNG &rngState) const override {
    viennaray::Vec3D<NumericType> origin{0.0, 0.0, height_};
    std::uniform_real_distribution<NumericType> distRadius(-radius_, radius_);
    origin[0] = distRadius(rngState);
    origin[1] = distRadius(rngState);
    NumericType dsq = origin[0] * origin[0] + origin[1] * origin[1];
    while (dsq > radius_ * radius_) {
      origin[0] = distRadius(rngState);
      origin[1] = distRadius(rngState);
      dsq = origin[0] * origin[0] + origin[1] * origin[1];
    }

    viennaray::Vec3D<NumericType> direction{0., 0., 0.};
    std::uniform_real_distribution<NumericType> uniDist;
    auto r1 = uniDist(rngState);
    auto tt = uniDist(rngState);
    NumericType sqrt1mtt = sqrtf(1 - tt);

    direction[2] = -sqrtf(tt);
    direction[0] = cosf(M_PI * 2.f * r1) * sqrt1mtt;
    direction[1] = sinf(M_PI * 2.f * r1) * sqrt1mtt;
    return {origin, direction};
  }
  size_t getNumPoints() const override { return 1; }
  NumericType getSourceArea() const override { return 1.; }
  NumericType getInitialRayWeight(const size_t idx) const { return 1.; }

private:
  const NumericType radius_ = 10.0;
  const NumericType height_ = 10.0;
};

template <int N>
std::array<NumericType, N> logSpace(NumericType start, NumericType end) {
  std::array<NumericType, N> result{};
  NumericType logStart = std::log10(start);
  NumericType logEnd = std::log10(end);
  NumericType step = (logEnd - logStart) / (N - 1);
  for (int i = 0; i < N; ++i) {
    result[i] = std::pow(10, logStart + i * step);
  }
  return result;
}

int main() {
  constexpr bool useGPU = false;
  auto lengths = logSpace<21>(.01, 10000.0);
  // std::array<NumericType, 1> lengths = {100.0};
  constexpr NumericType radius = 10.0;
  constexpr NumericType gridDelta = 1.0;
  constexpr int nRays = int(1e8);
  std::vector<unsigned> counts(4, 0);

  if constexpr (useGPU) {
    viennacore::CudaBuffer customDataBuffer;
    customDataBuffer.allocUpload(counts);

    auto context = viennacore::DeviceContext::createContext();
    viennaray::gpu::TraceTriangle<NumericType, D> tracer(context);

    viennaray::gpu::Particle<NumericType> particle;
    particle.name = "Particle";
    particle.dataLabels = {"TransmittedCount", "ReflectedCount",
                           "HitsToTransmission", "HitsToReflection"};

    std::unordered_map<std::string, unsigned int> pMap = {{"Particle", 0}};
    std::vector<viennaray::gpu::CallableConfig> cMap = {
        {0, viennaray::gpu::CallableSlot::COLLISION,
         "__direct_callable__transmissionTestCollision"},
        {0, viennaray::gpu::CallableSlot::REFLECTION,
         "__direct_callable__transmissionTestReflection"},
        {0, viennaray::gpu::CallableSlot::INIT,
         "__direct_callable__transmissionTestInit"}};

    tracer.setRngSeed(42);
    tracer.setCallables("TransmissionTest", context->modulePath);
    tracer.setParticleCallableMap({pMap, cMap});
    tracer.setNumberOfRaysFixed(nRays);
    tracer.insertNextParticle(particle);
    tracer.prepareParticlePrograms();

    std::ofstream logFile("transmission_log.txt");
    logFile << "LR\tTransmission\tReflection\tHitsToTransmission\tHitsToReflect"
               "ion\n";

    for (auto length : lengths) {
      std::cout << "Length: " << length << std::endl;
      std::cout << "L/R: " << length / radius << std::endl;
      auto [mesh, materialIds] = createCylinderMesh(gridDelta, radius, length);
      tracer.setGeometry(mesh, length);
      tracer.setMaterialIds(materialIds);

      tracer.apply();

      std::vector<float> results(mesh.triangles.size(), 0.f);
      for (int i = 0; i < 4; ++i) {
        counts[i] = 0;
        tracer.getFlux(results.data(), 0, i);
        for (auto &v : results) {
          counts[i] += static_cast<unsigned>(v);
        }
      }

      std::cout << "Bottom (transmission): " << counts[0] << std::endl;
      std::cout << "Top (reflection): " << counts[1] << std::endl;
      std::cout << "Transmission ratio: "
                << static_cast<double>(counts[0]) /
                       static_cast<double>(counts[0] + counts[1])
                << std::endl;

      logFile << length / radius << "\t" << counts[0] << "\t" << counts[1]
              << "\t" << counts[2] << "\t" << counts[3] << "\n";
    }

    logFile.close();
    customDataBuffer.free();
  } else {
    viennaray::TraceTriangle<NumericType, D> tracer;

    viennaray::BoundaryCondition rayBoundaryCondition[D];
    for (unsigned i = 0; i < D; ++i)
      rayBoundaryCondition[i] = viennaray::BoundaryCondition::IGNORE;

    tracer.setSourceDirection(viennaray::TraceDirection::POS_Z);
    tracer.setBoundaryConditions(rayBoundaryCondition);
    tracer.setNumberOfRaysFixed(nRays);
    tracer.setRngSeed(42);

    auto particle = std::make_unique<TransmissionParticle>();
    tracer.setParticleType(particle);

    std::ofstream logFile("transmission_log_CPU.txt");
    logFile << "LR\tTransmission\tReflection\tHitsToTransmission\tHitsToReflect"
               "ion\n";

    for (auto length : lengths) {
      std::cout << "Length: " << length << std::endl;
      std::cout << "L/R: " << length / radius << std::endl;
      auto [mesh, materialIds] = createCylinderMesh(gridDelta, radius, length);
      tracer.setGeometry(mesh);
      tracer.setMaterialIds(materialIds);

      auto source = std::make_shared<TransmissionSource>(radius, length);
      tracer.setSource(source);

      tracer.apply();

      auto &localData = tracer.getLocalData();
      counts[0] = static_cast<unsigned>(
          localData.getVectorData(0)[0]); // Transmission count
      counts[1] = static_cast<unsigned>(
          localData.getVectorData(0)[1]); // Reflection count

      std::cout << "Bottom (transmission): " << counts[0] << std::endl;
      std::cout << "Top (reflection): " << counts[1] << std::endl;
      std::cout << "Transmission ratio: "
                << static_cast<double>(counts[0]) /
                       static_cast<double>(counts[0] + counts[1])
                << std::endl;

      logFile << length / radius << "\t" << counts[0] << "\t" << counts[1]
              << "\t" << counts[2] << "\t" << counts[3] << "\n";
    }

    logFile.close();
  }

  return 0;
}