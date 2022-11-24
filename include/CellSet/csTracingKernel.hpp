#pragma once

#include <lsSmartPointer.hpp>

#include <rayBoundary.hpp>
#include <rayGeometry.hpp>
#include <rayRNG.hpp>
#include <raySource.hpp>
#include <rayUtil.hpp>

#include <csDenseCellSet.hpp>
#include <csTracePath.hpp>
#include <csTracingParticle.hpp>

template <typename T, int D> class csTracingKernel {
public:
  csTracingKernel(RTCDevice &pDevice, rayGeometry<T, D> &pRTCGeometry,
                  rayBoundary<T, D> &pRTCBoundary, raySource<T, D> &pSource,
                  std::unique_ptr<csAbstractParticle<T>> &pParticle,
                  const size_t pNumOfRayPerPoint, const size_t pNumOfRayFixed,
                  const bool pUseRandomSeed, const size_t pRunNumber,
                  lsSmartPointer<csDenseCellSet<T, D>> passedCellSet,
                  int passedExclude)
      : mDevice(pDevice), mGeometry(pRTCGeometry), mBoundary(pRTCBoundary),
        mSource(pSource), mParticle(pParticle->clone()),
        mNumRays(pNumOfRayFixed == 0
                     ? pSource.getNumPoints() * pNumOfRayPerPoint
                     : pNumOfRayFixed),
        mUseRandomSeeds(pUseRandomSeed), mRunNumber(pRunNumber),
        cellSet(passedCellSet), excludeMaterial(passedExclude),
        mGridDelta(cellSet->getGridDelta()) {
    assert(rtcGetDeviceProperty(mDevice, RTC_DEVICE_PROPERTY_VERSION) >=
               30601 &&
           "Error: The minimum version of Embree is 3.6.1");
  }

  void apply() {
    auto rtcScene = rtcNewScene(mDevice);
    rtcSetSceneFlags(rtcScene, RTC_SCENE_FLAG_NONE);
    auto bbquality = RTC_BUILD_QUALITY_HIGH;
    rtcSetSceneBuildQuality(rtcScene, bbquality);
    auto rtcGeometry = mGeometry.getRTCGeometry();
    auto rtcBoundary = mBoundary.getRTCGeometry();

    auto boundaryID = rtcAttachGeometry(rtcScene, rtcBoundary);
    auto geometryID = rtcAttachGeometry(rtcScene, rtcGeometry);
    assert(rtcGetDeviceError(mDevice) == RTC_ERROR_NONE &&
           "Embree device error");

    const csPair<T> meanFreePath = mParticle->getMeanFreePath();

    auto myCellSet = cellSet;

#pragma omp parallel shared(myCellSet)
    {
      rtcJoinCommitScene(rtcScene);

      alignas(128) auto rayHit =
          RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

      const int threadID = omp_get_thread_num();
      constexpr int numRngStates = 7;
      unsigned int seeds[numRngStates];
      if (mUseRandomSeeds) {
        std::random_device rd;
        for (size_t i = 0; i < numRngStates; ++i) {
          seeds[i] = static_cast<unsigned int>(rd());
        }
      } else {
        for (size_t i = 0; i < numRngStates; ++i) {
          seeds[i] = static_cast<unsigned int>((omp_get_thread_num() + 1) * 31 +
                                               i + mRunNumber);
        }
      }
      // It seems really important to use two separate seeds / states for
      // sampling the source and sampling reflections. When we use only one
      // state for both, then the variance is very high.
      rayRNG RngState1(seeds[0]);
      rayRNG RngState2(seeds[1]);
      rayRNG RngState3(seeds[2]);
      rayRNG RngState4(seeds[3]);
      rayRNG RngState5(seeds[4]);
      rayRNG RngState6(seeds[5]);
      rayRNG RngState7(seeds[6]);

      // thread-local particle object
      auto particle = mParticle->clone();

      // thread local path
      csTracePath<T> path;
      // if (!traceOnPath)
      path.useGridData(myCellSet->getNumberOfCells());

      auto rtcContext = RTCIntersectContext{};
      rtcInitIntersectContext(&rtcContext);

#pragma omp for schedule(dynamic)
      for (long long idx = 0; idx < mNumRays; ++idx) {
        particle->initNew(RngState6);

        mSource.fillRay(rayHit.ray, idx, RngState1, RngState2, RngState3,
                        RngState4); // fills also tnear

#ifdef VIENNARAY_USE_RAY_MASKING
        rayHit.ray.mask = -1;
#endif

        bool reflect = false;
        bool hitFromBack = false;
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
            mBoundary.processHit(rayHit, reflect);
            continue;
          }

          // Calculate point of impact
          const auto &ray = rayHit.ray;
          const rtcNumericType xx = ray.org_x + ray.dir_x * ray.tfar;
          const rtcNumericType yy = ray.org_y + ray.dir_y * ray.tfar;
          const rtcNumericType zz = ray.org_z + ray.dir_z * ray.tfar;

          /* -------- Hit from back -------- */
          const auto rayDir = rayTriple<T>{ray.dir_x, ray.dir_y, ray.dir_z};
          const auto geomNormal = mGeometry.getPrimNormal(rayHit.hit.primID);
          if (rayInternal::DotProduct(rayDir, geomNormal) > 0) {
            // If the dot product of the ray direction and the surface normal is
            // greater than zero, then we hit the back face of the disk.
            if (hitFromBack) {
              // if hitFromback == true, then the ray hits the back of a disk
              // the second time. In this case we ignore the ray.
              break;
            }
            hitFromBack = true;
            // Let ray through, i.e., continue.
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

          // get fill and reflection
          const auto fillnDirection =
              particle->surfaceHit(rayDir, geomNormal, reflect, RngState5);

          if (mGeometry.getMaterialId(rayHit.hit.primID) != excludeMaterial) {
            // trace in cell set
            auto hitPoint = std::array<T, 3>{xx, yy, zz};
            std::vector<csVolumeParticle<T>> particleStack;
            std::normal_distribution<T> normalDist{meanFreePath[0],
                                                   meanFreePath[1]};

            particleStack.emplace_back(csVolumeParticle<T>{
                hitPoint, rayDir, fillnDirection.first, 0., -1, 0});

            while (!particleStack.empty()) {
              auto volumeParticle = std::move(particleStack.back());
              particleStack.pop_back();

              // trace particle
              while (volumeParticle.energy >= 0) {
                volumeParticle.distance = -1;
                while (volumeParticle.distance < 0)
                  volumeParticle.distance = normalDist(RngState7);
                auto travelDist = csUtil::multNew(volumeParticle.direction,
                                                  volumeParticle.distance);
                csUtil::add(volumeParticle.position, travelDist);

                if (!checkBoundsPeriodic(volumeParticle.position))
                  break;

                auto newIdx = myCellSet->getIndex(volumeParticle.position);
                if (newIdx < 0)
                  break;

                if (newIdx != volumeParticle.cellId) {
                  volumeParticle.cellId = newIdx;
                  auto fill = particle->collision(volumeParticle, RngState7,
                                                  particleStack);
                  path.addGridData(newIdx, fill);
                }
              }
            }
          }

          if (!reflect) {
            break;
          }

          // Update ray direction and origin
#ifdef ARCH_X86
          reinterpret_cast<__m128 &>(rayHit.ray) =
              _mm_set_ps(1e-4f, zz, yy, xx);
          reinterpret_cast<__m128 &>(rayHit.ray.dir_x) =
              _mm_set_ps(0.0f, (rtcNumericType)fillnDirection.second[2],
                         (rtcNumericType)fillnDirection.second[1],
                         (rtcNumericType)fillnDirection.second[0]);
#else
          rayHit.ray.org_x = xx;
          rayHit.ray.org_y = yy;
          rayHit.ray.org_z = zz;
          rayHit.ray.tnear = 1e-4f;

          rayHit.ray.dir_x = (rtcNumericType)fillnDirection.second[0];
          rayHit.ray.dir_y = (rtcNumericType)fillnDirection.second[1];
          rayHit.ray.dir_z = (rtcNumericType)fillnDirection.second[2];
          rayHit.ray.time = 0.0f;
#endif
        } while (reflect);
#ifdef VIENNAPS_VERBOSE
        if (threadID == 0)
          printProgress(idx);
#endif
      } // end ray tracing for loop

#pragma omp critical
      myCellSet->mergePath(path, mNumRays);
    } // end parallel section

#ifdef VIENNAPS_VERBOSE
    std::cout << std::endl;
#endif

    rtcReleaseGeometry(rtcGeometry);
    rtcReleaseGeometry(rtcBoundary);
  }

private:
  bool checkBounds(const csTriple<T> &hitPoint) const {
    const auto &min = cellSet->getCellGrid()->minimumExtent;
    const auto &max = cellSet->getCellGrid()->maximumExtent;

    return hitPoint[0] >= min[0] && hitPoint[0] <= max[0] &&
           hitPoint[1] >= min[1] && hitPoint[1] <= max[1] &&
           hitPoint[2] >= min[2] && hitPoint[2] <= max[2];
  }

  bool checkBoundsPeriodic(csTriple<T> &hitPoint) const {
    const auto &min = cellSet->getCellGrid()->minimumExtent;
    const auto &max = cellSet->getCellGrid()->maximumExtent;

    if constexpr (D == 3) {
      if (hitPoint[2] < min[2] || hitPoint[2] > max[2])
        return false;

      if (hitPoint[0] < min[0]) {
        hitPoint[0] = max[0] - mGridDelta / 2.;
      } else if (hitPoint[0] > max[0]) {
        hitPoint[0] = min[0] + mGridDelta / 2.;
      }

      if (hitPoint[1] < min[1]) {
        hitPoint[1] = max[1] - mGridDelta / 2.;
      } else if (hitPoint[1] > max[1]) {
        hitPoint[1] = min[1] + mGridDelta / 2.;
      }
    } else {
      if (hitPoint[1] < min[1] || hitPoint[1] > max[1])
        return false;

      if (hitPoint[0] < min[0]) {
        hitPoint[0] = max[0] - mGridDelta / 2.;
      } else if (hitPoint[0] > max[0]) {
        hitPoint[0] = min[0] + mGridDelta / 2.;
      }
    }

    return true;
  }

private:
  RTCDevice &mDevice;
  rayGeometry<T, D> &mGeometry;
  rayBoundary<T, D> &mBoundary;
  raySource<T, D> &mSource;
  std::unique_ptr<csAbstractParticle<T>> const mParticle = nullptr;
  const long long mNumRays;
  const bool mUseRandomSeeds;
  const size_t mRunNumber;
  lsSmartPointer<csDenseCellSet<T, D>> cellSet = nullptr;
  const T mGridDelta = 0.;
  const int excludeMaterial = -1;

  void printProgress(size_t i) {
    float progress = static_cast<float>(i) / static_cast<float>(mNumRays);
    int barWidth = 70;

    std::cout << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
      if (i < pos)
        std::cout << "=";
      else if (i == pos)
        std::cout << ">";
      else
        std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
  }
};
