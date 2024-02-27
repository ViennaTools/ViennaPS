#pragma once

#include <psDomain.hpp>
#include <psUtils.hpp>
#include <rayBoundary.hpp>
#include <rayGeometry.hpp>
#include <rayRNG.hpp>
#include <rayTraceDirection.hpp>

#include <raySource.hpp>

template <typename NumericType, int D>
class CustomSource : public raySource<NumericType, D> {
public:
  CustomSource(std::array<NumericType, 3> pCoords, NumericType pCosinePower,
               std::array<int, 5> &pTraceSettings)
      : sourceBox(pCoords), rayDir(pTraceSettings[0]),
        firstDir(pTraceSettings[1]), secondDir(pTraceSettings[2]),
        minMax(pTraceSettings[3]), posNeg(pTraceSettings[4]),
        ee(((NumericType)2) / (pCosinePower + 1)) {}

  void fillRay(RTCRay &ray, const size_t idx, rayRNG &RngState) override final {
    auto origin = getOrigin(RngState);
    auto direction = getDirection(RngState);

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
  rayTriple<NumericType> getOrigin(rayRNG &RngState) {
    rayTriple<NumericType> origin{0., 0., 0.};
    std::uniform_real_distribution<NumericType> uniDist;
    auto r1 = uniDist(RngState);

    origin[rayDir] = sourceBox[2];
    origin[firstDir] = sourceBox[0] + (sourceBox[1] - sourceBox[0]) * r1;
    origin[secondDir] = 0.;

    return origin;
  }

  rayTriple<NumericType> getDirection(rayRNG &RngState) {
    rayTriple<NumericType> direction{0., 0., 0.};
    std::uniform_real_distribution<NumericType> uniDist;
    auto r1 = uniDist(RngState);
    auto r2 = uniDist(RngState);

    const NumericType tt = pow(r2, ee);
    direction[rayDir] = posNeg * sqrtf(tt);
    direction[firstDir] = cosf(M_PI * 2.f * r1) * sqrtf(1 - tt);

    if constexpr (D == 2) {
      direction[secondDir] = 0;
      rayInternal::Normalize(direction);
    } else {
      direction[secondDir] = sinf(M_PI * 2.f * r1) * sqrtf(1 - tt);
    }

    return direction;
  }

  const std::array<NumericType, 3> sourceBox;
  const int rayDir;
  const int firstDir;
  const int secondDir;
  const int minMax;
  const NumericType posNeg;
  const NumericType ee;
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
                         const int passedNumRaysPerPoint = 1000)
      : traceDevice(rtcNewDevice("hugepages=1")), domain(passedDomain),
        reflectionLimit(passedReflectionLimit),
        numRaysPerPoint(passedNumRaysPerPoint) {}
  ~psCalculateDiffusivity() {
    traceGeometry.releaseGeometry();
    rtcReleaseDevice(traceDevice);
  }

  void apply(NumericType minCoord, NumericType maxCoord, NumericType top,
             NumericType radius) {
    initMemoryFlags();
    initGeometry();

    auto boundingBox = traceGeometry.getBoundingBox();
    rayInternal::adjustBoundingBox<NumericType, D>(boundingBox, sourceDirection,
                                                   diskRadius);
    auto traceSettings = rayInternal::getTraceSettings(sourceDirection);
    auto boundary = rayBoundary<NumericType, D>(
        traceDevice, boundingBox, boundaryConditions, traceSettings);
    std::array<NumericType, 3> coords = {minCoord, maxCoord, top};
    auto raySource = CustomSource<NumericType, D>(coords, 1., traceSettings);

    auto result = runKernel(boundary, raySource);

    boundary.releaseGeometry();

    mesh->getCellData().insertNextScalarData(result.first, "Diffusivity");
    mesh->getCellData().insertNextScalarData(result.second, "NumHits");
    lsVTKWriter<NumericType>(mesh, "diffusivity.vtp").apply();

    auto &cellSet = domain->getCellSet();
    auto cellType = cellSet->getScalarData("CellType");
    auto diff = cellSet->addScalarData("Diffusivity", 0.);
    auto cells = cellSet->getElements();
    auto numCells = cells.size();
    auto &points = mesh->getNodes();

    for (unsigned i = 0; i < numCells; ++i) {
      auto cellCenter = cellSet->getCellCenter(i);
      if (cellType->at(i) != 1. || cellCenter[D - 1] > top) {
        continue;
      }
      NumericType distSum = 0.;
      for (unsigned j = 0; j < points.size(); j++) {
        auto distance = rayInternal::Distance(cellCenter, points[j]);
        if (distance < radius) {
          distSum += distance;
          diff->at(i) += distance * result.first[j];
        }
      }
      diff->at(i) /= distSum;
    }
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
    for (unsigned i = 0; i < D; ++i)
      boundaryConditions[i] =
          convertBoundaryCondition(domain->getGrid().getBoundaryConditions(i));
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

  rayBoundaryCondition convertBoundaryCondition(
      lsBoundaryConditionEnum<D> originalBoundaryCondition) const {
    switch (originalBoundaryCondition) {
    case lsBoundaryConditionEnum<D>::REFLECTIVE_BOUNDARY:
      return rayBoundaryCondition::REFLECTIVE;

    case lsBoundaryConditionEnum<D>::INFINITE_BOUNDARY:
      return rayBoundaryCondition::IGNORE;

    case lsBoundaryConditionEnum<D>::PERIODIC_BOUNDARY:
      return rayBoundaryCondition::PERIODIC;

    case lsBoundaryConditionEnum<D>::POS_INFINITE_BOUNDARY:
      return rayBoundaryCondition::IGNORE;

    case lsBoundaryConditionEnum<D>::NEG_INFINITE_BOUNDARY:
      return rayBoundaryCondition::IGNORE;
    }
    return rayBoundaryCondition::IGNORE;
  }

  std::pair<std::vector<NumericType>, std::vector<NumericType>>
  runKernel(rayBoundary<NumericType, D> &boundary,
            raySource<NumericType, D> &source) {
    auto rtcScene = rtcNewScene(traceDevice);
    rtcSetSceneFlags(rtcScene, RTC_SCENE_FLAG_NONE);
    auto bbquality = RTC_BUILD_QUALITY_HIGH;
    rtcSetSceneBuildQuality(rtcScene, bbquality);
    auto rtcGeometry = traceGeometry.getRTCGeometry();
    auto rtcBoundary = boundary.getRTCGeometry();

    auto boundaryID = rtcAttachGeometry(rtcScene, rtcBoundary);
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

        source.fillRay(rayHit.ray, idx, RngState); // fills also tnear

#ifdef VIENNARAY_USE_RAY_MASKING
        rayHit.ray.mask = -1;
#endif
        bool reflect = false;
        bool hitFromBack = false;
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

          /* -------- Boundary hit -------- */
          if (rayHit.hit.geomID == boundaryID) {
            boundary.processHit(rayHit, reflect);
            continue;
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
          data[rayHit.hit.primID] += rayHit.ray.tfar;
          hitCount[rayHit.hit.primID]++;

          // check for additional intersections
          for (const auto &id :
               traceGeometry.getNeighborIndicies(rayHit.hit.primID)) {
            rtcNumericType distance;
            if (checkLocalIntersection(ray, id, distance)) {
              data[id] += rayHit.ray.tfar;
              hitCount[id]++;
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

    diskAreas = computeDiskAreas(boundary);

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
    for (unsigned i = 0; i < traceGeometry.getNumPoints(); ++i) {
      result[i] /= hitCounts[i] * diskAreas[i];
    }

    rtcReleaseScene(rtcScene);

    return std::pair<std::vector<NumericType>, std::vector<NumericType>>{
        result, hitCounts};
  }

  std::vector<NumericType>
  computeDiskAreas(rayBoundary<NumericType, D> &boundary) {
    constexpr double eps = 1e-3;
    auto bdBox = traceGeometry.getBoundingBox();
    const auto numOfPrimitives = traceGeometry.getNumPoints();
    const auto boundaryDirs = boundary.getDirs();
    auto areas = std::vector<NumericType>(numOfPrimitives, 0);

#pragma omp parallel for
    for (long idx = 0; idx < numOfPrimitives; ++idx) {
      auto const &disk = traceGeometry.getPrimRef(idx);

      if constexpr (D == 3) {
        areas[idx] = disk[3] * disk[3] * M_PI; // full disk area

        if (std::fabs(disk[boundaryDirs[0]] - bdBox[0][boundaryDirs[0]]) <
                eps ||
            std::fabs(disk[boundaryDirs[0]] - bdBox[1][boundaryDirs[0]]) <
                eps) {
          // disk intersects boundary in first direction
          areas[idx] /= 2;
        }

        if (std::fabs(disk[boundaryDirs[1]] - bdBox[0][boundaryDirs[1]]) <
                eps ||
            std::fabs(disk[boundaryDirs[1]] - bdBox[1][boundaryDirs[1]]) <
                eps) {
          // disk intersects boundary in second direction
          areas[idx] /= 2;
        }
      } else {
        areas[idx] = 2 * disk[3];
        auto normal = traceGeometry.getNormalRef(idx);

        // test min boundary
        if (std::abs(disk[boundaryDirs[0]] - bdBox[0][boundaryDirs[0]]) <
            disk[3]) {
          NumericType insideTest =
              1 - normal[boundaryDirs[0]] * normal[boundaryDirs[0]];
          if (insideTest > 1e-4) {
            insideTest =
                std::abs(disk[boundaryDirs[0]] - bdBox[0][boundaryDirs[0]]) /
                std::sqrt(insideTest);
            if (insideTest < disk[3]) {
              areas[idx] -= disk[3] - insideTest;
            }
          }
        }

        // test max boundary
        if (std::abs(disk[boundaryDirs[0]] - bdBox[1][boundaryDirs[0]]) <
            disk[3]) {
          NumericType insideTest =
              1 - normal[boundaryDirs[0]] * normal[boundaryDirs[0]];
          if (insideTest > 1e-4) {
            insideTest =
                std::abs(disk[boundaryDirs[0]] - bdBox[1][boundaryDirs[0]]) /
                std::sqrt(insideTest);
            if (insideTest < disk[3]) {
              areas[idx] -= disk[3] - insideTest;
            }
          }
        }
      }
    }
    return areas;
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

private:
  const unsigned int reflectionLimit = 100;
  NumericType gridDelta = 0;
  NumericType diskRadius = 0;
  rayTraceDirection sourceDirection =
      D == 3 ? rayTraceDirection::POS_Z : rayTraceDirection::POS_Y;
  rayBoundaryCondition boundaryConditions[D] = {};
  unsigned int seed = 15235135;
  long long numRays = 0;
  long numRaysPerPoint = 1000;
};