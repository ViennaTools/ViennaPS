#pragma once

#include <geometries/psMakeHole.hpp>
#include <geometries/psMakeTrench.hpp>
#include <models/psIonBeamEtching.hpp>
#include <psDomain.hpp>
#include <rayParticle.hpp>

#define MAKE_GEO Trench
#define DEFAULT_GRID_DELTA 0.1
#define DEFAULT_STICKING 0.1
#define DIM 2
#define FIXED_RAYS false

constexpr int particleType = 1;
using TranslatorType = std::unordered_map<unsigned long, unsigned long>;
using namespace viennaps;

template <class NumericType>
auto Trench(NumericType gridDelta = DEFAULT_GRID_DELTA) {
  NumericType xExtent = 20.;
  NumericType yExtent = 20.;
  NumericType width = 10.;
  NumericType depth = 50.;

  auto domain = Domain<NumericType, DIM>::New(
      gridDelta, xExtent, yExtent, BoundaryType::REFLECTIVE_BOUNDARY);
  MakeTrench<NumericType, DIM>(domain, width, depth).apply();
  return domain;
}

template <class NumericType>
auto Hole(NumericType gridDelta = DEFAULT_GRID_DELTA) {
  NumericType xExtent = 10.;
  NumericType yExtent = 10.;
  NumericType radius = 3.0;
  NumericType depth = 30.;

  auto domain = Domain<NumericType, DIM>::New(
      gridDelta, xExtent, yExtent, BoundaryType::REFLECTIVE_BOUNDARY);
  MakeHole<NumericType, DIM>(domain, radius, depth).apply();
  return domain;
}

template <class NumericType, int N>
std::array<NumericType, N> linspace(NumericType start, NumericType end) {
  std::array<NumericType, N> arr{};
  NumericType step = (end - start) / static_cast<NumericType>(N - 1);
  for (int i = 0; i < N; ++i) {
    arr[i] = start + i * step;
  }
  return arr;
}

template <typename NumericType> auto getIBEParameters() {
  viennaps::IBEParameters<NumericType> params;
  params.redepositionRate = 1.0;
  params.cos4Yield.isDefined = true;
  params.cos4Yield.a1 = 0.1;
  params.cos4Yield.a2 = 0.2;
  params.cos4Yield.a3 = 0.3;
  params.cos4Yield.a4 = 0.4;
  params.thetaRMin = 0.0;
  params.thetaRMax = 90.0;
  params.meanEnergy = 250.0;
  params.sigmaEnergy = 10.0;
  params.thresholdEnergy = 20.0;
  params.n_l = 10;
  params.inflectAngle = 80.0;
  params.minAngle = 85.0;
  params.exponent = 100.0;
  return params;
}

template <typename NumericType, int D> auto makeCPUParticle() {
  if constexpr (particleType == 0) {
    auto particle =
        std::make_unique<viennaray::DiffuseParticle<NumericType, D>>(
            DEFAULT_STICKING, "flux");
    return particle;
  } else if constexpr (particleType == 1) {
    auto params = getIBEParameters<NumericType>();
    auto particle = std::make_unique<
        viennaps::impl::IBEIonWithRedeposition<NumericType, D>>(params);
    return particle;
  } else {
    static_assert(particleType == 0 || particleType == 1,
                  "Unsupported particle type!");
  }
}

template <typename NumericType, int D>
std::tuple<viennaray::gpu::Particle<NumericType>,
           std::unordered_map<std::string, unsigned>,
           std::vector<viennaray::gpu::CallableConfig>>
makeGPUParticle() {
  if constexpr (particleType == 0) {
    auto particle = viennaray::gpu::Particle<NumericType>();
    particle.name = "SingleParticle";
    particle.sticking = DEFAULT_STICKING;
    particle.dataLabels.emplace_back("flux");
    std::unordered_map<std::string, unsigned> pMap = {{"SingleParticle", 0}};
    std::vector<viennaray::gpu::CallableConfig> cMap = {
        {0, viennaray::gpu::CallableSlot::COLLISION,
         "__direct_callable__singleNeutralCollision"},
        {0, viennaray::gpu::CallableSlot::REFLECTION,
         "__direct_callable__singleNeutralReflection"}};
    return {particle, pMap, cMap};
  } else if constexpr (particleType == 1) {
    auto params_ = getIBEParameters<NumericType>();
    viennaray::gpu::Particle<NumericType> particle{
        .name = "IBEIon", .cosineExponent = params_.exponent};
    particle.dataLabels.push_back("flux");
    if (params_.redepositionRate > 0.) {
      particle.dataLabels.push_back(
          viennaps::impl::IBESurfaceModel<NumericType>::redepositionLabel);
    }
    std::unordered_map<std::string, unsigned> pMap = {{"IBEIon", 0}};
    std::vector<viennaray::gpu::CallableConfig> cMap = {
        {0, viennaray::gpu::CallableSlot::COLLISION,
         "__direct_callable__IBECollision"},
        {0, viennaray::gpu::CallableSlot::REFLECTION,
         "__direct_callable__IBEReflection"},
        {0, viennaray::gpu::CallableSlot::INIT, "__direct_callable__IBEInit"}};
    return {particle, pMap, cMap};
  } else {
    static_assert(particleType == 0 || particleType == 1,
                  "Unsupported particle type!");
  }
}

auto getDeviceParams() {
  if constexpr (particleType != 1) {
    throw std::runtime_error(
        "getDeviceParams is only defined for particleType 1 (IBEIon)!");
  }
  auto params_ = getIBEParameters<float>();
  // Parameters to upload to device
  viennaps::gpu::impl::IonParams deviceParams;
  deviceParams.thetaRMin = viennaps::constants::degToRad(params_.thetaRMin);
  deviceParams.thetaRMax = viennaps::constants::degToRad(params_.thetaRMax);
  deviceParams.meanEnergy = params_.meanEnergy;
  deviceParams.sigmaEnergy = params_.sigmaEnergy;
  deviceParams.thresholdEnergy =
      std::sqrt(params_.thresholdEnergy); // precompute sqrt
  deviceParams.minAngle = viennaps::constants::degToRad(params_.minAngle);
  deviceParams.inflectAngle =
      viennaps::constants::degToRad(params_.inflectAngle);
  deviceParams.n_l = params_.n_l;
  deviceParams.B_sp = 0.f; // not used in IBE
  if (params_.cos4Yield.isDefined) {
    deviceParams.a1 = params_.cos4Yield.a1;
    deviceParams.a2 = params_.cos4Yield.a2;
    deviceParams.a3 = params_.cos4Yield.a3;
    deviceParams.a4 = params_.cos4Yield.a4;
    deviceParams.aSum = params_.cos4Yield.aSum();
  }
  deviceParams.redepositionRate = params_.redepositionRate;
  deviceParams.redepositionThreshold = params_.redepositionThreshold;

  return deviceParams;
}

template <class NumericType, int D, class TracerType>
void setupTriangleGeometry(
    SmartPointer<Domain<NumericType, D>> &domain,
    SmartPointer<viennals::Mesh<float>> &surfaceMesh_,
    SmartPointer<KDTree<NumericType, Vec3D<NumericType>>> &elementKdTree_,
    TracerType &rayTracer_) {
  CreateSurfaceMesh<NumericType, float, D>(
      domain->getLevelSets().back(), surfaceMesh_, elementKdTree_, 1e-12, 0.05)
      .apply();

  viennaray::TriangleMesh triangleMesh;

  if constexpr (D == 2) {
    viennaray::LineMesh lineMesh;
    lineMesh.gridDelta = static_cast<float>(domain->getGridDelta());
    lineMesh.lines = surfaceMesh_->lines;
    lineMesh.nodes = surfaceMesh_->nodes;
    lineMesh.minimumExtent = surfaceMesh_->minimumExtent;
    lineMesh.maximumExtent = surfaceMesh_->maximumExtent;

    triangleMesh = convertLinesToTriangles(lineMesh);
    assert(triangleMesh.triangles.size() > 0);

    std::vector<Vec3D<NumericType>> triangleCenters;
    triangleCenters.reserve(triangleMesh.triangles.size());
    for (const auto &tri : triangleMesh.triangles) {
      Vec3D<NumericType> center = {0, 0, 0};
      for (int i = 0; i < 3; ++i) {
        center[0] += triangleMesh.nodes[tri[i]][0];
        center[1] += triangleMesh.nodes[tri[i]][1];
        center[2] += triangleMesh.nodes[tri[i]][2];
      }
      triangleCenters.push_back(center / static_cast<NumericType>(3.0));
    }
    assert(triangleCenters.size() > 0);
    elementKdTree_->setPoints(triangleCenters);
    elementKdTree_->build();
  } else {
    triangleMesh = CreateTriangleMesh(
        static_cast<float>(domain->getGridDelta()), surfaceMesh_);
  }

  rayTracer_.setGeometry(triangleMesh);

  if constexpr (D == 2) {
    surfaceMesh_->nodes = std::move(triangleMesh.nodes);
    surfaceMesh_->triangles = std::move(triangleMesh.triangles);
    surfaceMesh_->getCellData().insertReplaceVectorData(
        std::move(triangleMesh.normals), "Normals");
    surfaceMesh_->minimumExtent = triangleMesh.minimumExtent;
    surfaceMesh_->maximumExtent = triangleMesh.maximumExtent;
  }
}

template <class NumericType, int D, class TracerType>
void setupLineGeometry(
    SmartPointer<Domain<NumericType, D>> &domain,
    SmartPointer<viennals::Mesh<float>> &surfaceMesh_,
    SmartPointer<KDTree<NumericType, Vec3D<NumericType>>> &elementKdTree_,
    TracerType &rayTracer_) {
  CreateSurfaceMesh<NumericType, float, D>(
      domain->getLevelSets().back(), surfaceMesh_, elementKdTree_, 1e-12, 0.05)
      .apply();

  viennaray::LineMesh lineMesh(surfaceMesh_->nodes, surfaceMesh_->lines,
                               static_cast<float>(domain->getGridDelta()));
  // lines might have changed, so we need to update the surfaceMesh_ later

  std::vector<Vec3D<NumericType>> elementCenters(lineMesh.lines.size());
  for (int i = 0; i < lineMesh.lines.size(); ++i) {
    auto const &p0 = lineMesh.nodes[lineMesh.lines[i][0]];
    auto const &p1 = lineMesh.nodes[lineMesh.lines[i][1]];
    auto center = (p0 + p1) / 2.f;
    elementCenters[i] = Vec3D<NumericType>{static_cast<NumericType>(center[0]),
                                           static_cast<NumericType>(center[1]),
                                           static_cast<NumericType>(center[2])};
  }
  elementKdTree_->setPoints(elementCenters);
  elementKdTree_->build();

  rayTracer_.setGeometry(lineMesh);

  surfaceMesh_->nodes = std::move(lineMesh.nodes);
  surfaceMesh_->lines = std::move(lineMesh.lines);
  surfaceMesh_->getCellData().insertReplaceVectorData(
      std::move(lineMesh.normals), "Normals");
  surfaceMesh_->minimumExtent = lineMesh.minimumExtent;
  surfaceMesh_->maximumExtent = lineMesh.maximumExtent;
}

template <typename NumericType, class TracerType>
void postProcessLineData(
    viennals::PointData<NumericType> &pointData,
    SmartPointer<viennals::Mesh<NumericType>> diskMesh, int smoothingNeighbors,
    NumericType gridDelta, TracerType &rayTracer_,
    SmartPointer<KDTree<NumericType, Vec3D<NumericType>>> elementKdTree_,
    SmartPointer<viennals::Mesh<float>> surfaceMesh_) {

  const auto numRates = rayTracer_.getNumberOfRates();
  const auto numPoints = rayTracer_.getNumberOfElements();
  const auto numDisks = diskMesh->nodes.size();
  assert(numRates > 0);
  auto particles = rayTracer_.getParticles();
  auto const &elementNormals =
      *surfaceMesh_->getCellData().getVectorData("Normals");
  auto const &normals = *diskMesh->getCellData().getVectorData("Normals");
  const auto numElements = surfaceMesh_->lines.size();

  NumericType conversionRadius = gridDelta * (smoothingNeighbors + 1);
  conversionRadius *= conversionRadius; // use squared radius

  std::vector<std::vector<std::pair<unsigned, NumericType>>> elementsToPoint;
  elementsToPoint.reserve(numDisks);

  for (int i = 0; i < numDisks; i++) {
    auto closePoints =
        elementKdTree_
            ->findNearestWithinRadius(diskMesh->nodes[i], conversionRadius)
            .value();

    std::vector<std::pair<unsigned, NumericType>> closePointsArray;
    std::vector<NumericType> weights(closePoints.size(), NumericType(0));

    unsigned numClosePoints = 0;
    for (std::size_t n = 0; n < closePoints.size(); ++n) {
      const auto &p = closePoints[n];
      assert(p.first < numElements);

      NumericType weight = 0;
      for (int k = 0; k < 3; ++k) {
        weight += normals[i][k] * elementNormals[p.first][k];
      }
      weight = std::max(weight, NumericType(0));

      if (weight > NumericType(1e-6) && !std::isnan(weight)) {
        weights[n] = weight;
        ++numClosePoints;
      }
    }

    if (numClosePoints == 0) { // fallback to nearest point
      auto nearestPoint = elementKdTree_->findNearest(diskMesh->nodes[i]);
      closePointsArray.emplace_back(static_cast<unsigned>(nearestPoint->first),
                                    NumericType(1));
    }

    // Compute weighted average
    const NumericType sum =
        std::accumulate(weights.begin(), weights.end(), NumericType(0));

    if (sum > NumericType(0)) {
      for (std::size_t k = 0; k < closePoints.size(); ++k) {
        if (weights[k] > NumericType(0)) {
          closePointsArray.emplace_back(
              static_cast<unsigned>(closePoints[k].first), weights[k] / sum);
        }
      }
    } else {
      // Fallback if all weights were discarded
      auto nearestPoint = elementKdTree_->findNearest(diskMesh->nodes[i]);
      closePointsArray.emplace_back(static_cast<unsigned>(nearestPoint->first),
                                    NumericType(1));
    }

    elementsToPoint.push_back(closePointsArray);
  }
  assert(elementsToPoint.size() == numDisks);

  for (int pIdx = 0; pIdx < particles.size(); pIdx++) {
    for (int dIdx = 0; dIdx < particles[pIdx].dataLabels.size(); dIdx++) {
      auto elementFlux = rayTracer_.getFlux(pIdx, dIdx, smoothingNeighbors);
      auto name = particles[pIdx].dataLabels[dIdx];

      // convert line fluxes to disk fluxes
      std::vector<NumericType> diskFlux(numDisks, 0.);

      for (int i = 0; i < numDisks; i++) {
        for (const auto &elemPair : elementsToPoint[i]) {
          diskFlux[i] += static_cast<NumericType>(elementFlux[elemPair.first]) *
                         elemPair.second;
        }
      }

      pointData.insertReplaceScalarData(std::move(diskFlux), name);
    }
  }
}