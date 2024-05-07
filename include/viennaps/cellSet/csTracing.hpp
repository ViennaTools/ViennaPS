#pragma once

#include "csDenseCellSet.hpp"
#include "csPointSource.hpp"
#include "csTracingKernel.hpp"
#include "csTracingParticle.hpp"

#include <lsToDiskMesh.hpp>
#include <rayBoundary.hpp>
#include <rayGeometry.hpp>
#include <rayParticle.hpp>
#include <raySourceRandom.hpp>
#include <rayUtil.hpp>

template <class T, int D> class csTracing {
private:
  lsSmartPointer<csDenseCellSet<T, D>> cellSet = nullptr;
  std::unique_ptr<csAbstractParticle<T>> mParticle = nullptr;

  RTCDevice mDevice;
  rayGeometry<T, D> mGeometry;
  size_t mNumberOfRaysPerPoint = 0;
  size_t mNumberOfRaysFixed = 1000;
  T mGridDelta = 0;
  rayBoundaryCondition mBoundaryConditions[D] = {};
  const rayTraceDirection mSourceDirection =
      D == 2 ? rayTraceDirection::POS_Y : rayTraceDirection::POS_Z;
  bool mUseRandomSeeds = true;
  bool usePrimaryDirection = false;
  rayTriple<T> primaryDirection = {0.};
  size_t mRunNumber = 0;
  int excludeMaterialId = -1;
  bool usePointSource = false;
  csTriple<T> pointSourceOrigin = {0.};
  csTriple<T> pointSourceDirection = {0.};

public:
  csTracing() : mDevice(rtcNewDevice("hugepages=1")) {
    // TODO: currently only periodic boundary conditions are implemented in
    // csTracingKernel
    for (int i = 0; i < D; i++)
      mBoundaryConditions[i] = rayBoundaryCondition::PERIODIC;
  }

  ~csTracing() {
    mGeometry.releaseGeometry();
    rtcReleaseDevice(mDevice);
  }

  void apply() {
    createGeometry();
    initMemoryFlags();
    auto boundingBox = mGeometry.getBoundingBox();
    rayInternal::adjustBoundingBox<T, D>(
        boundingBox, mSourceDirection, mGridDelta * rayInternal::DiskFactor<D>);
    auto traceSettings = rayInternal::getTraceSettings(mSourceDirection);

    auto boundary = rayBoundary<T, D>(mDevice, boundingBox, mBoundaryConditions,
                                      traceSettings);

    std::array<rayTriple<T>, 3> orthoBasis;
    if (usePrimaryDirection) {
      orthoBasis = rayInternal::getOrthonormalBasis(primaryDirection);
      csLogger::getInstance()
          .addInfo("Using primary direction: " +
                   std::to_string(primaryDirection[0]) + " " +
                   std::to_string(primaryDirection[1]) + " " +
                   std::to_string(primaryDirection[2]))
          .print();
    }

    if (usePointSource) {
      auto raySource =
          csPointSource<T, D>(pointSourceOrigin, pointSourceDirection,
                              traceSettings, mGeometry.getNumPoints());

      csTracingKernel(mDevice, mGeometry, boundary, raySource, mParticle,
                      mNumberOfRaysPerPoint, mNumberOfRaysFixed,
                      mUseRandomSeeds, mRunNumber++, cellSet,
                      excludeMaterialId - 1)
          .apply();
    } else {
      auto raySource = raySourceRandom<T, D>(
          boundingBox, mParticle->getSourceDistributionPower(), traceSettings,
          mGeometry.getNumPoints(), usePrimaryDirection, orthoBasis);

      csTracingKernel(mDevice, mGeometry, boundary, raySource, mParticle,
                      mNumberOfRaysPerPoint, mNumberOfRaysFixed,
                      mUseRandomSeeds, mRunNumber++, cellSet,
                      excludeMaterialId - 1)
          .apply();
    }

    averageNeighborhood();
    boundary.releaseGeometry();
  }

  void setCellSet(lsSmartPointer<csDenseCellSet<T, D>> passedCellSet) {
    cellSet = passedCellSet;
  }

  void setPointSource(const csTriple<T> &passedOrigin,
                      const csTriple<T> &passedDirection) {
    usePointSource = true;
    pointSourceOrigin = passedOrigin;
    pointSourceDirection = passedDirection;
  }

  template <typename ParticleType>
  void setParticle(std::unique_ptr<ParticleType> &p) {
    static_assert(std::is_base_of<csAbstractParticle<T>, ParticleType>::value &&
                  "Particle object does not interface correct class");
    mParticle = p->clone();
  }

  void setTotalNumberOfRays(const size_t passedNumber) {
    mNumberOfRaysFixed = passedNumber;
    mNumberOfRaysPerPoint = 0;
  }

  void setNumberOfRaysPerPoint(const size_t passedNumber) {
    mNumberOfRaysPerPoint = passedNumber;
    mNumberOfRaysFixed = 0;
  }

  void setPrimaryDirection(const rayTriple<T> pPrimaryDirection) {
    primaryDirection = pPrimaryDirection;
    usePrimaryDirection = true;
  }

  void setExcludeMaterialId(int passedId) { excludeMaterialId = passedId; }

  lsSmartPointer<csDenseCellSet<T, D>> getCellSet() const { return cellSet; }

  void averageNeighborhood() {
    auto data = cellSet->getFillingFractions();
    auto materialIds = cellSet->getScalarData("Material");
    const auto &elems = cellSet->getElements();
    const auto &nodes = cellSet->getNodes();
    std::vector<T> average(data->size(), 0.);

#pragma omp parallel for
    for (int i = 0; i < data->size(); i++) {
      if (data->at(i) < 0) {
        average[i] = -1.;
        continue;
      }
      if (materialIds->at(i) == excludeMaterialId)
        continue;

      int numNeighbors = 1;
      average[i] += data->at(i);

      for (int d = 0; d < D; d++) {
        auto mid = calcMidPoint(nodes[elems[i][0]]);
        mid[d] -= mGridDelta;
        auto elemId = cellSet->getIndex(mid);
        if (elemId >= 0) {
          if (data->at(elemId) >= 0 &&
              materialIds->at(elemId) != excludeMaterialId) {
            average[i] += data->at(elemId);
            numNeighbors++;
          }
        }

        mid[d] += 2 * mGridDelta;
        elemId = cellSet->getIndex(mid);
        if (elemId >= 0) {
          if (data->at(elemId) >= 0 &&
              materialIds->at(elemId) != excludeMaterialId) {
            average[i] += data->at(elemId);
            numNeighbors++;
          }
        }
      }
      average[i] /= static_cast<T>(numNeighbors);
    }

#pragma omp parallel for
    for (int i = 0; i < data->size(); i++) {
      data->at(i) = average[i];
    }
  }

  void averageNeighborhoodSingleMaterial(int materialId) {
    auto data = cellSet->getFillingFractions();
    auto materialIds = cellSet->getScalarData("Material");
    const auto &elems = cellSet->getElements();
    const auto &nodes = cellSet->getNodes();
    std::vector<T> average(data->size(), 0.);

#pragma omp parallel for
    for (int i = 0; i < data->size(); i++) {
      if (materialIds->at(i) != materialId)
        continue;

      int numNeighbors = 1;
      average[i] += data->at(i);
      for (int d = 0; d < D; d++) {
        auto mid = calcMidPoint(nodes[elems[i][0]]);
        mid[d] -= mGridDelta;
        auto elemId = cellSet->getIndex(mid);
        if (elemId >= 0) {
          if (data->at(elemId) >= 0 && materialIds->at(elemId) == materialId) {
            average[i] += data->at(elemId);
            numNeighbors++;
          }
        }

        mid[d] += 2 * mGridDelta;
        elemId = cellSet->getIndex(mid);
        if (elemId >= 0) {
          if (data->at(elemId) >= 0 && materialIds->at(elemId) == materialId) {
            average[i] += data->at(elemId);
            numNeighbors++;
          }
        }
      }
      average[i] /= static_cast<T>(numNeighbors);
    }

#pragma omp parallel for
    for (int i = 0; i < data->size(); i++) {
      data->at(i) = average[i];
    }
  }

private:
  void createGeometry() {
    auto levelSets = cellSet->getLevelSets();
    auto diskMesh = lsSmartPointer<lsMesh<T>>::New();
    lsToDiskMesh<T, D> converter(diskMesh);
    for (auto ls : *levelSets) {
      converter.insertNextLevelSet(ls);
    }
    converter.apply();
    auto points = diskMesh->getNodes();
    auto normals = *diskMesh->getCellData().getVectorData("Normals");
    auto materialIds = *diskMesh->getCellData().getScalarData("MaterialIds");
    mGridDelta = levelSets->back()->getGrid().getGridDelta();
    mGeometry.initGeometry(mDevice, points, normals,
                           mGridDelta * rayInternal::DiskFactor<D>);
    mGeometry.setMaterialIds(materialIds);
  }

  inline csTriple<T> calcMidPoint(const csTriple<T> &minNode) {
    return csTriple<T>{minNode[0] + mGridDelta / T(2),
                       minNode[1] + mGridDelta / T(2),
                       minNode[2] + mGridDelta / T(2)};
  }

  void initMemoryFlags() {
#ifdef ARCH_X86
    // for best performance set FTZ and DAZ flags in MXCSR control and status
    // register
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif
  }
};
