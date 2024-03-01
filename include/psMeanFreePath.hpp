#pragma once

#include <psDomain.hpp>
#include <psKDTree.hpp>
#include <psLogger.hpp>
#include <psUtils.hpp>

#include <rayGeometry.hpp>
#include <rayRNG.hpp>
#include <rayReflection.hpp>

template <typename NumericType, int D> class CustomSource {
public:
  CustomSource(psSmartPointer<csDenseCellSet<NumericType, D>> passCellSet,
               const NumericType passedCutoff)
      : cellSet(passCellSet), numCells(cellSet->getElements().size()),
        materialIds(cellSet->getScalarData("Material")), cutoff(passedCutoff) {}

  void fillRay(RTCRay &ray, const size_t idx, rayRNG &RngState) {
    std::uniform_real_distribution<NumericType> uniDist;
    std::uniform_int_distribution<unsigned> cellDist(0, numCells - 1);

    unsigned cellIdx = cellDist(RngState);
    auto material = materialIds->at(cellIdx);
    auto origin = cellSet->getCellCenter(cellIdx);
    while (!psMaterialMap::isMaterial(material, psMaterial::GAS) ||
           origin[D - 1] > cutoff) {
      cellIdx = cellDist(RngState);
      material = materialIds->at(cellIdx);
      origin = cellSet->getCellCenter(cellIdx);
    }

    rayTriple<NumericType> direction{0., 0., 0.};

    for (int i = 0; i < D; ++i) {
      direction[i] = uniDist(RngState) * 2 - 1;
    }

    rayInternal::Normalize(direction);

#ifdef ARCH_X86
    reinterpret_cast<__m128 &>(ray) =
        _mm_set_ps(1e-4f, (float)origin[2], (float)origin[1], (float)origin[0]);

    reinterpret_cast<__m128 &>(ray.dir_x) = _mm_set_ps(
        0.0f, (float)direction[2], (float)direction[1], (float)direction[0]);
#else
    ray.org_x = (float)origin[0];
    ray.org_y = (float)origin[1];
    ray.org_z = (float)origin[2];
    ray.tnear = 1e-4f;

    ray.dir_x = (float)direction[0];
    ray.dir_y = (float)direction[1];
    ray.dir_z = (float)direction[2];
    ray.time = 0.0f;
#endif
  }

private:
  psSmartPointer<csDenseCellSet<NumericType, D>> cellSet = nullptr;
  const std::size_t numCells;
  const NumericType cutoff = 0.;
  std::vector<NumericType> *materialIds;
};

template <class NumericType, int D> class psMeanFreePath {

public:
  psMeanFreePath() : traceDevice(rtcNewDevice("hugepages=1")) {
    static_assert(D == 2 && "Diffusivity calculation only implemented for 2D");
  }

  ~psMeanFreePath() {
    traceGeometry.releaseGeometry();
    rtcReleaseDevice(traceDevice);
  }

  void setDomain(const psSmartPointer<psDomain<NumericType, D>> passedDomain) {
    domain = passedDomain;
  }

  void setBulkLambda(const NumericType passedBulkLambda) {
    bulkLambda = passedBulkLambda;
  }

  void setMaterial(const psMaterial passedMaterial) {
    material = passedMaterial;
  }

  void setTopCutoff(const NumericType passedTop) { top = passedTop; }

  void setNumNeighbors(const unsigned int passedNumNeighbors) {
    numNeighbors = passedNumNeighbors;
  }

  void setNumRaysPerPoint(const int passedNumRaysPerPoint) {
    numRaysPerPoint = passedNumRaysPerPoint;
  }

  void setReflectionLimit(const int passedReflectionLimit) {
    reflectionLimit = passedReflectionLimit;
  }

  NumericType getMaxLambda() const { return maxLambda; }

  void apply() {
    psLogger::getInstance().addInfo("Calculating mean free path ...").print();
    initGeometry();
    auto result = runKernel();
    interpolateResult(result);
  }

private:
  void interpolateResult(
      std::pair<std::vector<NumericType>, std::vector<NumericType>> &result) {
    auto &cellSet = domain->getCellSet();
    auto cellType = cellSet->getScalarData("CellType");
    if (!cellType) {
      psLogger::getInstance()
          .addWarning("Cell set not segmented. Segmentation required.")
          .print();
      return;
    }
    auto lambda = cellSet->addScalarData("MeanFreePath", 0.);
    auto numCells = cellSet->getElements().size();
    auto &points = mesh->getNodes();

    psKDTree<NumericType, std::array<NumericType, 3>> kdTree;
    kdTree.setPoints(points);
    kdTree.build();

    maxLambda = bulkLambda;
    for (unsigned i = 0; i < numCells; ++i) {
      auto cellCenter = cellSet->getCellCenter(i);
      if (cellType->at(i) != 1.) {
        continue;
      }

      if (cellCenter[D - 1] > top) {
        lambda->at(i) = bulkLambda;
        continue;
      }

      auto neighbors = kdTree.findKNearest(cellCenter, numNeighbors);

      if (!neighbors) {
        psLogger::getInstance()
            .addWarning("No neighbors found for cell " + std::to_string(i))
            .print();
        continue;
      }

      NumericType distanceSum = 0.;
      for (const auto &n : neighbors.value()) {
        auto &point = points[n.first];
        if (point[D - 1] >= top)
          continue;

        distanceSum += n.second; // distance
        lambda->at(i) += n.second * result.first[n.first];
      }
      lambda->at(i) /= distanceSum;

      if (lambda->at(i) > maxLambda) {
        maxLambda = lambda->at(i);
      }
    }
  }

  void initGeometry() {
    mesh = lsSmartPointer<lsMesh<NumericType>>::New();
    lsToDiskMesh<NumericType, D>(domain->getLevelSets()->back(), mesh).apply();
    auto &points = mesh->getNodes();
    auto normals = mesh->getCellData().getVectorData("Normals");
    gridDelta = domain->getGrid().getGridDelta();
    assert(normals);
    diskRadius = gridDelta * rayInternal::DiskFactor<D>;
    traceGeometry.initGeometry(traceDevice, points, *normals, diskRadius);
    numRays = points.size() * numRaysPerPoint;
  }

  std::pair<std::vector<NumericType>, std::vector<NumericType>> runKernel() {
#ifdef ARCH_X86
    // for best performance set FTZ and DAZ flags in MXCSR control and status
    // register
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif
    auto source = CustomSource<NumericType, D>(domain->getCellSet(), top);
    auto topGeometry = rayGeometry<NumericType, D>();
    {
      auto boundingBox = domain->getBoundingBox();
      NumericType minx = boundingBox[0][0];
      NumericType maxx = boundingBox[1][0];
      auto topPoints = std::vector<rayTriple<NumericType>>();
      auto topNormals = std::vector<rayTriple<NumericType>>();
      rayTriple<NumericType> normal = {0., 0., 0.};
      normal[D - 1] = -1.;
      while (minx < maxx) {
        rayTriple<NumericType> point = {minx, top + gridDelta, 0.};
        topPoints.push_back(point);
        topNormals.push_back(normal);
        minx += gridDelta;
      }
      topGeometry.initGeometry(traceDevice, topPoints, topNormals, gridDelta);
    }

    auto rtcScene = rtcNewScene(traceDevice);
    rtcSetSceneFlags(rtcScene, RTC_SCENE_FLAG_NONE);
    auto bbquality = RTC_BUILD_QUALITY_HIGH;
    rtcSetSceneBuildQuality(rtcScene, bbquality);
    auto rtcGeometry = traceGeometry.getRTCGeometry();
    auto geometryID = rtcAttachGeometry(rtcScene, rtcGeometry);
    auto rtcTopGeometry = topGeometry.getRTCGeometry();
    auto sourceID = rtcAttachGeometry(rtcScene, rtcTopGeometry);
    assert(rtcGetDeviceError(traceDevice) == RTC_ERROR_NONE &&
           "Embree device error");

    // thread local data storage
    const int numThreads = omp_get_max_threads();
    std::vector<std::vector<NumericType>> threadLocalData(numThreads);
    std::vector<std::vector<unsigned>> threadLocalHitCount(numThreads);

#pragma omp parallel
    {
      rtcJoinCommitScene(rtcScene);

      alignas(128) auto rayHit =
          RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

      const int threadNum = omp_get_thread_num();
      auto &data = threadLocalData[threadNum];
      data.resize(traceGeometry.getNumPoints(), 0.);
      auto &hitCount = threadLocalHitCount[threadNum];
      hitCount.resize(traceGeometry.getNumPoints(), 0);

      auto rtcContext = RTCIntersectContext{};
      rtcInitIntersectContext(&rtcContext);

#pragma omp for schedule(dynamic)
      for (long long idx = 0; idx < numRays; ++idx) {
        // particle specific RNG seed
        auto particleSeed = rayInternal::tea<3>(idx, seed);
        rayRNG RngState(particleSeed);

        source.fillRay(rayHit.ray, idx, RngState);

#ifdef VIENNARAY_USE_RAY_MASKING
        rayHit.ray.mask = -1;
#endif
        std::vector<unsigned> sourcePrimIDs;
        unsigned numReflections = 0;
        while (true) {
          rayHit.ray.tfar = std::numeric_limits<rtcNumericType>::max();
          rayHit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
          rayHit.hit.geomID = RTC_INVALID_GEOMETRY_ID;

          // Run the intersection
          rtcIntersect1(rtcScene, &rtcContext, &rayHit);

          /* -------- No hit -------- */
          if (rayHit.hit.geomID == RTC_INVALID_GEOMETRY_ID) {
            break;
          }

          /* -------- Source hit -------- */
          if (rayHit.hit.geomID == sourceID) {
            for (const auto &id : sourcePrimIDs) {
              data[id] += rayHit.ray.tfar + bulkLambda;
              hitCount[id]++;
            }
            break;
          }

          /* -------- Geometry hit -------- */
          const auto &ray = rayHit.ray;
          const auto rayDir =
              rayTriple<NumericType>{ray.dir_x, ray.dir_y, ray.dir_z};
          auto geomNormal = traceGeometry.getPrimNormal(rayHit.hit.primID);

          /* -------- Hit at disk backside -------- */
          if (rayInternal::DotProduct(rayDir, geomNormal) > 0) {
            break;
          }

          for (const auto &id : sourcePrimIDs) {
            data[id] += rayHit.ray.tfar;
            hitCount[id]++;
          }

          sourcePrimIDs.clear();
          sourcePrimIDs.push_back(rayHit.hit.primID);
          // check for additional intersections
          for (const auto &id :
               traceGeometry.getNeighborIndicies(rayHit.hit.primID)) {
            rtcNumericType distance;
            if (checkLocalIntersection(ray, id, distance)) {
              sourcePrimIDs.push_back(id);
            }
          }

          // Reflect
          ++numReflections;
          if (numReflections >= reflectionLimit) {
            break;
          }

          auto reflectedDirection =
              rayReflectionDiffuse<NumericType, D>(geomNormal, RngState);

          const rtcNumericType xx = ray.org_x + ray.dir_x * ray.tfar;
          const rtcNumericType yy = ray.org_y + ray.dir_y * ray.tfar;
          const rtcNumericType zz = ray.org_z + ray.dir_z * ray.tfar;
#ifdef ARCH_X86
          reinterpret_cast<__m128 &>(rayHit.ray) =
              _mm_set_ps(1e-4f, zz, yy, xx);
          reinterpret_cast<__m128 &>(rayHit.ray.dir_x) =
              _mm_set_ps(0.0f, (rtcNumericType)reflectedDirection[2],
                         (rtcNumericType)reflectedDirection[1],
                         (rtcNumericType)reflectedDirection[0]);
#else
          rayHit.ray.org_x = xx;
          rayHit.ray.org_y = yy;
          rayHit.ray.org_z = zz;
          rayHit.ray.tnear = 1e-4f;

          rayHit.ray.dir_x = (rtcNumericType)reflectedDirection[0];
          rayHit.ray.dir_y = (rtcNumericType)reflectedDirection[1];
          rayHit.ray.dir_z = (rtcNumericType)reflectedDirection[2];
          rayHit.ray.time = 0.0f;
#endif
        }
      }
    }

    topGeometry.releaseGeometry();
    rtcReleaseScene(rtcScene);

    // reduce data
    std::vector<NumericType> result(traceGeometry.getNumPoints(), 0);
    for (const auto &data : threadLocalData) {
      for (unsigned i = 0; i < traceGeometry.getNumPoints(); ++i) {
        result[i] += data[i];
      }
    }

    // reduce hit counts
    std::vector<NumericType> hitCounts(traceGeometry.getNumPoints(), 0);
    for (const auto &data : threadLocalHitCount) {
      for (unsigned i = 0; i < traceGeometry.getNumPoints(); ++i) {
        hitCounts[i] += data[i];
      }
    }

    // normalize data
#pragma omp parallel for
    for (unsigned i = 0; i < traceGeometry.getNumPoints(); ++i) {
      if (hitCounts[i] > 0)
        result[i] /= hitCounts[i];
      else
        result[i] = -1;
    }

    // smooth result
    auto oldResult = result;
#pragma omp parallel for
    for (size_t idx = 0; idx < traceGeometry.getNumPoints(); idx++) {
      auto neighborhood = traceGeometry.getNeighborIndicies(idx);
      for (auto const &nbi : neighborhood) {
        result[idx] += oldResult[nbi];
      }
      result[idx] /= (neighborhood.size() + 1);
    }

    return std::pair<std::vector<NumericType>, std::vector<NumericType>>{
        result, hitCounts};
  }

  bool checkLocalIntersection(RTCRay const &ray, const unsigned int primID,
                              rtcNumericType &impactDistance) {
    auto const &rayOrigin =
        *reinterpret_cast<rayTriple<rtcNumericType> const *>(&ray.org_x);
    auto const &rayDirection =
        *reinterpret_cast<rayTriple<rtcNumericType> const *>(&ray.dir_x);

    const auto &normal = traceGeometry.getNormalRef(primID);
    const auto &disk = traceGeometry.getPrimRef(primID);
    const auto &diskOrigin =
        *reinterpret_cast<rayTriple<rtcNumericType> const *>(&disk);

    auto prodOfDirections = rayInternal::DotProduct(normal, rayDirection);
    if (prodOfDirections > 0.f) {
      return false;
    }

    constexpr auto eps = 1e-6f;
    if (std::fabs(prodOfDirections) < eps) {
      return false;
    }

    auto ddneg = rayInternal::DotProduct(diskOrigin, normal);
    auto tt =
        (ddneg - rayInternal::DotProduct(normal, rayOrigin)) / prodOfDirections;
    if (tt <= 0) {
      return false;
    }

    auto rayDirectionC = rayTriple<rtcNumericType>{
        rayDirection[0], rayDirection[1], rayDirection[2]};
    rayInternal::Scale(tt, rayDirectionC);
    auto hitpoint = rayInternal::Sum(rayOrigin, rayDirectionC);
    auto distance = rayInternal::Distance(hitpoint, diskOrigin);
    auto const &radius = disk[3];
    if (radius > distance) {
      impactDistance = distance;
      return true;
    }
    return false;
  }

private:
  psSmartPointer<psDomain<NumericType, D>> domain = nullptr;
  psSmartPointer<lsMesh<NumericType>> mesh = nullptr;
  rayGeometry<NumericType, D> traceGeometry;
  RTCDevice traceDevice;

  NumericType top = 0;
  NumericType bulkLambda = 0;
  NumericType meanThermalVelocity = 1.;
  NumericType gridDelta = 0;
  NumericType diskRadius = 0;
  NumericType maxLambda = 0.;
  unsigned int numNeighbors = 10;
  unsigned int seed = 15235135;
  unsigned int reflectionLimit = 100;
  long long numRays = 0;
  long numRaysPerPoint = 1000;
  psMaterial material = psMaterial::GAS;
};