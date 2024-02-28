#pragma once

#include <psDomain.hpp>
#include <psKDTree.hpp>
#include <psUtils.hpp>

#include <rayGeometry.hpp>
#include <rayRNG.hpp>
#include <rayReflection.hpp>

template <typename NumericType, int D> class CustomSource {
public:
  CustomSource(psSmartPointer<csDenseCellSet<NumericType, D>> passCellSet,
               const NumericType passedCutoff)
      : cellSet(passCellSet), numCells(cellSet->getElements().size()),
        materialIds(cellSet->getScalarData("Material")) {}

  void fillRay(RTCRay &ray, const size_t idx, rayRNG &RngState) {
    std::uniform_real_distribution<NumericType> uniDist;

    unsigned cellIdx = static_cast<unsigned>(uniDist(RngState) * numCells);
    auto material = materialIds->at(cellIdx);
    auto origin = cellSet->getCellCenter(cellIdx);
    while (!psMaterialMap::isMaterial(material, psMaterial::GAS) ||
           origin[D - 2] > cutoff) {
      cellIdx = static_cast<unsigned>(uniDist(RngState) * numCells);
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

template <class NumericType, int D> class psCalculateDiffusivity {
  psSmartPointer<psDomain<NumericType, D>> domain = nullptr;
  psSmartPointer<lsMesh<NumericType>> mesh = nullptr;
  rayGeometry<NumericType, D> traceGeometry;
  RTCDevice traceDevice;

public:
  psCalculateDiffusivity() : traceDevice(rtcNewDevice("hugepages=1")) {}
  psCalculateDiffusivity(psSmartPointer<psDomain<NumericType, D>> passedDomain,
                         const int passedReflectionLimit = 100,
                         const int passedNumRaysPerPoint = 1000,
                         const psMaterial passedMaterial = psMaterial::GAS)
      : traceDevice(rtcNewDevice("hugepages=1")), domain(passedDomain),
        reflectionLimit(passedReflectionLimit),
        numRaysPerPoint(passedNumRaysPerPoint), material(passedMaterial) {}
  ~psCalculateDiffusivity() {
    traceGeometry.releaseGeometry();
    rtcReleaseDevice(traceDevice);
  }

  NumericType apply(NumericType top, int numNeighbors) {
    initMemoryFlags();
    initGeometry();

    auto result = runKernel(top);

    mesh->getCellData().insertNextScalarData(result.first, "MeanFreePath");
    mesh->getCellData().insertNextScalarData(result.second, "NumHits");
    lsVTKWriter<NumericType>(mesh, "diffusivity.vtp").apply();

    auto &cellSet = domain->getCellSet();
    auto cellType = cellSet->getScalarData("CellType");
    auto diff = cellSet->addScalarData("MeanFreePath", 0.);
    auto numCells = cellSet->getElements().size();
    auto &points = mesh->getNodes();

    psKDTree<NumericType, std::array<NumericType, 3>> kdTree;
    kdTree.setPoints(points);
    kdTree.build();

    NumericType maxDiffusivity = -1.;
    for (unsigned i = 0; i < numCells; ++i) {
      auto cellCenter = cellSet->getCellCenter(i);
      if (cellType->at(i) != 1. || cellCenter[D - 1] > top) {
        continue;
      }

      auto neighbors = kdTree.findKNearest(cellCenter, numNeighbors);

      if (!neighbors) {
        std::cout << "No neighbors found for cell " << i << std::endl;
        continue;
      }

      NumericType distanceSum = 0.;
      for (const auto &n : neighbors.value()) {
        auto &point = points[n.first];
        if (point[D - 1] >= top)
          continue;

        distanceSum += n.second; // distance
        diff->at(i) += n.second * result.first[n.first];
      }
      diff->at(i) /= distanceSum;

      if (diff->at(i) > maxDiffusivity) {
        maxDiffusivity = diff->at(i);
      }
    }

    return maxDiffusivity;
  }

private:
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

  void initMemoryFlags() {
#ifdef ARCH_X86
    // for best performance set FTZ and DAZ flags in MXCSR control and status
    // register
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif
  }

  std::pair<std::vector<NumericType>, std::vector<NumericType>>
  runKernel(const NumericType top) {
    auto source = CustomSource<NumericType, D>(domain->getCellSet(), top);

    auto rtcScene = rtcNewScene(traceDevice);
    rtcSetSceneFlags(rtcScene, RTC_SCENE_FLAG_NONE);
    auto bbquality = RTC_BUILD_QUALITY_HIGH;
    rtcSetSceneBuildQuality(rtcScene, bbquality);
    auto rtcGeometry = traceGeometry.getRTCGeometry();
    auto geometryID = rtcAttachGeometry(rtcScene, rtcGeometry);
    assert(rtcGetDeviceError(traceDevice) == RTC_ERROR_NONE &&
           "Embree device error");

    // thread local data storage
    const int numThreads = omp_get_max_threads();
    std::vector<std::vector<NumericType>> threadLocalData(numThreads);
    std::vector<std::vector<unsigned>> threadLocalHitCount(numThreads);
    std::vector<NumericType> diskAreas;

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
        if (threadNum == 0) {
          psUtils::printProgress(idx, numRays);
        }

        // particle specific RNG seed
        auto particleSeed = rayInternal::tea<3>(idx, seed);
        rayRNG RngState(particleSeed);

        source.fillRay(rayHit.ray, idx, RngState);

#ifdef VIENNARAY_USE_RAY_MASKING
        rayHit.ray.mask = -1;
#endif
        bool reflect = false;
        bool hitFromBack = false;
        std::vector<unsigned> sourcePrimIDs;
        unsigned numReflections = 0;
        do {
          rayHit.ray.tfar = std::numeric_limits<rtcNumericType>::max();
          rayHit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
          rayHit.hit.geomID = RTC_INVALID_GEOMETRY_ID;

          // Run the intersection
          rtcIntersect1(rtcScene, &rtcContext, &rayHit);

          /* -------- No hit -------- */
          if (rayHit.hit.geomID == RTC_INVALID_GEOMETRY_ID) {
            reflect = false;
            break;
          }

          // Calculate point of impact
          const auto &ray = rayHit.ray;
          const rtcNumericType xx = ray.org_x + ray.dir_x * ray.tfar;
          const rtcNumericType yy = ray.org_y + ray.dir_y * ray.tfar;
          const rtcNumericType zz = ray.org_z + ray.dir_z * ray.tfar;

          /* -------- Hit from back -------- */
          const auto rayDir =
              rayTriple<NumericType>{ray.dir_x, ray.dir_y, ray.dir_z};
          const auto geomNormal =
              traceGeometry.getPrimNormal(rayHit.hit.primID);
          if (rayInternal::DotProduct(rayDir, geomNormal) > 0) {
            if (hitFromBack) {
              break;
            }
            hitFromBack = true;
            reflect = true;
#ifdef ARCH_X86
            reinterpret_cast<__m128 &>(rayHit.ray) =
                _mm_set_ps(1e-4f, zz, yy, xx);
#else
            rayHit.ray.org_x = xx;
            rayHit.ray.org_y = yy;
            rayHit.ray.org_z = zz;
            rayHit.ray.tnear = 1e-4f;
#endif
            // keep ray direction as it is
            continue;
          }

          /* -------- Surface hit -------- */
          assert(rayHit.hit.geomID == geometryID && "Geometry hit ID invalid");
          if (!sourcePrimIDs.empty()) {
            for (const auto &id : sourcePrimIDs) {
              data[id] += rayHit.ray.tfar;
              hitCount[id]++;
            }
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
            reflect = false;
            break;
          }

          auto reflectedDirection =
              rayReflectionDiffuse<NumericType, D>(geomNormal, RngState);

          // Update ray direction and origin
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
        } while (reflect);
      }
    }

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

    smoothResult(result);

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
      // Disk normal is pointing away from the ray direction,
      // i.e., this might be a hit from the back or no hit at all.
      return false;
    }

    constexpr auto eps = 1e-6f;
    if (std::fabs(prodOfDirections) < eps) {
      // Ray is parallel to disk surface
      return false;
    }

    // TODO: Memoize ddneg
    auto ddneg = rayInternal::DotProduct(diskOrigin, normal);
    auto tt =
        (ddneg - rayInternal::DotProduct(normal, rayOrigin)) / prodOfDirections;
    if (tt <= 0) {
      // Intersection point is behind or exactly on the ray origin.
      return false;
    }

    // copy ray direction
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

  void smoothResult(std::vector<NumericType> &flux) {
    assert(flux.size() == traceGeometry.getNumPoints() &&
           "Unequal number of points in smoothResult");
    auto oldFlux = flux;
#pragma omp parallel for
    for (size_t idx = 0; idx < traceGeometry.getNumPoints(); idx++) {
      auto neighborhood = traceGeometry.getNeighborIndicies(idx);
      for (auto const &nbi : neighborhood) {
        flux[idx] += oldFlux[nbi];
      }
      flux[idx] /= (neighborhood.size() + 1);
    }
  }

private:
  const unsigned int reflectionLimit = 100;
  NumericType gridDelta = 0;
  NumericType diskRadius = 0;
  unsigned int seed = 15235135;
  long long numRays = 0;
  long numRaysPerPoint = 1000;
  const psMaterial material = psMaterial::GAS;
};