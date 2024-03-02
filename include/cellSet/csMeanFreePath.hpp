#pragma once

#include <psDomain.hpp>
#include <psKDTree.hpp>
#include <psLogger.hpp>
#include <psUtils.hpp>

#include <rayGeometry.hpp>
#include <rayRNG.hpp>
#include <rayReflection.hpp>

template <class NumericType, int D> class csMeanFreePath {

public:
  csMeanFreePath() : traceDevice(rtcNewDevice("hugepages=1")) {
    static_assert(D == 2 &&
                  "Mean free path calculation only implemented for 2D");
  }

  ~csMeanFreePath() { rtcReleaseDevice(traceDevice); }

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

  void setNumRaysPerCell(const int passedNumRaysPerCell) {
    numRaysPerCell = passedNumRaysPerCell;
  }

  NumericType getMaxLambda() const { return maxLambda; }

  void apply() {
    psLogger::getInstance().addInfo("Calculating mean free path ...").print();
    domain->getCellSet()->addScalarData("MeanFreePath", 0.);
    runKernel();
  }

private:
  void runKernel() {
#ifdef ARCH_X86
    // for best performance set FTZ and DAZ flags in MXCSR control and status
    // register
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif
    auto &cellSet = domain->getCellSet();
    unsigned numCells = cellSet->getElements().size();
    auto data = cellSet->getScalarData("MeanFreePath");
    auto materials = cellSet->getScalarData("Material");

    auto traceGeometry = rayGeometry<NumericType, D>();
    {
      auto mesh = lsSmartPointer<lsMesh<NumericType>>::New();
      lsToDiskMesh<NumericType, D>(domain->getLevelSets()->back(), mesh)
          .apply();
      auto &points = mesh->getNodes();
      auto normals = mesh->getCellData().getVectorData("Normals");
      gridDelta = domain->getGrid().getGridDelta();
      traceGeometry.initGeometry(traceDevice, points, *normals,
                                 gridDelta * rayInternal::DiskFactor<D>);
    }

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

    maxLambda = 0.;

#pragma omp parallel reduction(max : maxLambda)
    {
      rtcJoinCommitScene(rtcScene);

      alignas(128) auto rayHit =
          RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

      auto rtcContext = RTCIntersectContext{};
      rtcInitIntersectContext(&rtcContext);

#pragma omp for
      for (unsigned idx = 0; idx < numCells; ++idx) {
        if (!psMaterialMap::isMaterial(materials->at(idx), material))
          continue;

        auto cellCenter = cellSet->getCellCenter(idx);
        if (cellCenter[D - 1] > top) {
          data->at(idx) = bulkLambda;
          continue;
        }

        auto &ray = rayHit.ray;
#ifdef ARCH_X86
        reinterpret_cast<__m128 &>(ray) =
            _mm_set_ps(1e-4f, (float)cellCenter[2], (float)cellCenter[1],
                       (float)cellCenter[0]);
#else
        ray.org_x = (float)cellCenter[0];
        ray.org_y = (float)cellCenter[1];
        ray.org_z = (float)cellCenter[2];
        ray.tnear = 1e-4f;
#endif

#ifdef VIENNARAY_USE_RAY_MASKING
        ray.mask = -1;
#endif
        for (unsigned cIdx = 0; cIdx < numRaysPerCell; ++cIdx) {

          auto direction = getDirection(cIdx);

#ifdef ARCH_X86
          reinterpret_cast<__m128 &>(ray.dir_x) =
              _mm_set_ps(0.0f, (float)direction[2], (float)direction[1],
                         (float)direction[0]);
#else
          ray.dir_x = (float)direction[0];
          ray.dir_y = (float)direction[1];
          ray.dir_z = (float)direction[2];
          ray.time = 0.0f;
#endif

          ray.tfar = std::numeric_limits<rtcNumericType>::max();
          rayHit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
          rayHit.hit.geomID = RTC_INVALID_GEOMETRY_ID;

          // Run the intersection
          rtcIntersect1(rtcScene, &rtcContext, &rayHit);

          /* -------- No hit -------- */
          if (rayHit.hit.geomID == RTC_INVALID_GEOMETRY_ID) {
            continue;
          }

          /* -------- Source hit -------- */
          if (rayHit.hit.geomID == sourceID) {
            data->at(idx) += ray.tfar + bulkLambda;
            continue;
          }

          /* -------- Geometry hit -------- */
          const auto rayDir =
              rayTriple<NumericType>{ray.dir_x, ray.dir_y, ray.dir_z};
          auto geomNormal = traceGeometry.getPrimNormal(rayHit.hit.primID);

          /* -------- Hit at disk backside -------- */
          if (rayInternal::DotProduct(rayDir, geomNormal) > 0) {
            continue;
          }

          data->at(idx) += ray.tfar;
        }

        data->at(idx) /= numRaysPerCell;
        maxLambda = std::max(maxLambda, data->at(idx));
      }
    } // end of parallel section

    traceGeometry.releaseGeometry();
    topGeometry.releaseGeometry();
    rtcReleaseScene(rtcScene);
  }

  std::array<NumericType, 3> getDirection(const unsigned int idx) {
    std::array<NumericType, 3> direction;
    NumericType theta = idx * 2. * M_PI / numRaysPerCell;
    direction[0] = std::cos(theta);
    direction[1] = std::sin(theta);
    direction[2] = 0.;
    return direction;
  }

private:
  psSmartPointer<psDomain<NumericType, D>> domain = nullptr;
  RTCDevice traceDevice;

  NumericType top = 0;
  NumericType gridDelta = 0;
  NumericType bulkLambda = 0;
  NumericType maxLambda = 0.;
  long numRaysPerCell = 100;
  psMaterial material = psMaterial::GAS;
};