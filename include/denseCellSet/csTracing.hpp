#pragma once

#include <embree3/rtcore.h>
#include <lsSmartPointer.hpp>
#include <lsToDiskMesh.hpp>
#include <rayGeometry.hpp>
#include <rayParticle.hpp>
#include <raySourceRandom.hpp>
#include <rayUtil.hpp>

#include "csDenseCellSet.hpp"
#include "csTracingKernel.hpp"
#include "csTracingParticle.hpp"

template <class T, int D> class csTracing {
private:
  lsSmartPointer<csDenseCellSet<T, D>> cellSet = nullptr;
  std::unique_ptr<csAbstractParticle<T>> mParticle = nullptr;

  RTCDevice mDevice;
  rayGeometry<T, D> mGeometry;
  size_t mNumberOfRaysPerPoint = 0;
  size_t mNumberOfRaysFixed = 1000;
  T mGridDelta = 0;
  rayTraceBoundary mBoundaryConds[D] = {rayTraceBoundary::PERIODIC,
                                        rayTraceBoundary::PERIODIC,
                                        rayTraceBoundary::PERIODIC};
  rayTraceDirection mSourceDirection = rayTraceDirection::POS_Z;
  bool mUseRandomSeeds = true;
  size_t mRunNumber = 0;
  int excludeMaterialId = -1;
  bool traceOnPath = true;

public:
  csTracing() : mDevice(rtcNewDevice("hugepages=1")) {}

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

    auto boundary =
        rayBoundary<T, D>(mDevice, boundingBox, mBoundaryConds, traceSettings);

    auto raySource = raySourceRandom<T, D>(
        boundingBox, mParticle->getSourceDistributionPower(), traceSettings,
        mGeometry.getNumPoints());

    auto tracer = csTracingKernel<T, D>(
        mDevice, mGeometry, boundary, raySource, mParticle,
        mNumberOfRaysPerPoint, mNumberOfRaysFixed, mUseRandomSeeds,
        mRunNumber++, cellSet, mGridDelta, traceOnPath, excludeMaterialId - 1);
    tracer.apply();

    averageNeighborhood();
    boundary.releaseGeometry();
  }

  void setCellSet(lsSmartPointer<csDenseCellSet<T, D>> passedCellSet) {
    cellSet = passedCellSet;
  }

  template <typename ParticleType>
  void setParticle(std::unique_ptr<ParticleType> &p) {
    static_assert(std::is_base_of<csAbstractParticle<T>, ParticleType>::value &&
                  "Particle object does not interface correct class");
    mParticle = p->clone();
  }

  lsSmartPointer<csDenseCellSet<T, D>> getCellSet() const { return cellSet; }

  void writeVTU(std::string fileName) { cellSet->writeVTU(fileName); }

  void setTotalNumberOfRays(const size_t passedNumber) {
    mNumberOfRaysFixed = passedNumber;
    mNumberOfRaysPerPoint = 0;
  }

  void setNumberOfRaysPerPoint(const size_t passedNumber) {
    mNumberOfRaysPerPoint = passedNumber;
    mNumberOfRaysFixed = 0;
  }

  void setExcludeMaterialId(int passedId) { excludeMaterialId = passedId; }

  void setTraceOnPath(bool passedTrace) { traceOnPath = passedTrace; }

  void averageNeighborhood() {
    auto data = cellSet->getFillingFractions();
    auto materialIds = cellSet->getMaterialIds();
    std::vector<T> average(data->size(), 0.);

#pragma omp parallel for
    for (size_t i = 0; i < data->size(); i++) {
      if (data->at(i) < 0) {
        average[i] = -1.;
        continue;
      }
      if (materialIds->at(i) == excludeMaterialId)
        continue;

      int numNeighbors = 0;
      for (const auto n : cellSet->getNeighbors(i)) {
        if (data->at(n) >= 0 && materialIds->at(i) != excludeMaterialId) {
          average[i] += data->at(n);
          numNeighbors++;
        }
      }
      average[i] /= static_cast<T>(numNeighbors);
    }

#pragma omp parallel for
    for (size_t i = 0; i < data->size(); i++) {
      data->at(i) = average[i];
    }
  }

private:
  void createGeometry() {
    auto surfaceLevelSet = cellSet->getSurface();
    auto levelSets = cellSet->getLevelSets();
    auto diskMesh = lsSmartPointer<lsMesh<>>::New();
    lsToDiskMesh<T, D> converter(diskMesh);
    for (auto ls : *levelSets) {
      converter.insertNextLevelSet(ls);
    }
    converter.apply();
    auto points = diskMesh->getNodes();
    auto normals = *diskMesh->getCellData().getVectorData("Normals");
    mGridDelta = surfaceLevelSet->getGrid().getGridDelta();
    auto materialIds = *diskMesh->getCellData().getScalarData("MaterialIds");
    mGeometry.initGeometry(mDevice, points, normals,
                           mGridDelta * rayInternal::DiskFactor<D>);
    mGeometry.setMaterialIds(materialIds);
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