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

namespace viennacs {

using namespace viennacore;

template <class T, int D> class Tracing {
private:
  SmartPointer<DenseCellSet<T, D>> cellSet = nullptr;
  std::unique_ptr<AbstractParticle<T>> mParticle = nullptr;

  RTCDevice mDevice;
  viennaray::Geometry<T, D> mGeometry;
  size_t mNumberOfRaysPerPoint = 0;
  size_t mNumberOfRaysFixed = 1000;
  T mGridDelta = 0;
  viennaray::BoundaryCondition mBoundaryConditions[D] = {};
  const viennaray::TraceDirection mSourceDirection =
      D == 2 ? viennaray::TraceDirection::POS_Y
             : viennaray::TraceDirection::POS_Z;
  bool mUseRandomSeeds = true;
  bool usePrimaryDirection = false;
  Vec3D<T> primaryDirection = {0.};
  size_t mRunNumber = 0;
  int excludeMaterialId = -1;
  bool usePointSource = false;
  Vec3D<T> pointSourceOrigin = {0.};
  Vec3D<T> pointSourceDirection = {0.};

public:
  Tracing() : mDevice(rtcNewDevice("hugepages=1")) {
    // TODO: currently only periodic boundary conditions are implemented in
    // TracingKernel
    for (int i = 0; i < D; i++)
      mBoundaryConditions[i] = viennaray::BoundaryCondition::PERIODIC;
  }

  ~Tracing() {
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

    auto boundary = viennaray::Boundary<T, D>(
        mDevice, boundingBox, mBoundaryConditions, traceSettings);

    std::array<Vec3D<T>, 3> orthoBasis;
    if (usePrimaryDirection) {
      orthoBasis = rayInternal::getOrthonormalBasis(primaryDirection);
      Logger::getInstance()
          .addInfo("Using primary direction: " +
                   std::to_string(primaryDirection[0]) + " " +
                   std::to_string(primaryDirection[1]) + " " +
                   std::to_string(primaryDirection[2]))
          .print();
    }

    if (usePointSource) {
      auto raySource = std::make_shared<PointSource<T, D>>(
          pointSourceOrigin, pointSourceDirection, traceSettings,
          mGeometry.getNumPoints());

      TracingKernel<T, D>(mDevice, mGeometry, boundary, std::move(raySource),
                          mParticle, mNumberOfRaysPerPoint, mNumberOfRaysFixed,
                          mUseRandomSeeds, mRunNumber++, cellSet,
                          excludeMaterialId - 1)
          .apply();
    } else {
      auto raySource = std::make_shared<viennaray::SourceRandom<T, D>>(
          boundingBox, mParticle->getSourceDistributionPower(), traceSettings,
          mGeometry.getNumPoints(), usePrimaryDirection, orthoBasis);

      TracingKernel<T, D>(mDevice, mGeometry, boundary, std::move(raySource),
                          mParticle, mNumberOfRaysPerPoint, mNumberOfRaysFixed,
                          mUseRandomSeeds, mRunNumber++, cellSet,
                          excludeMaterialId - 1)
          .apply();
    }

    averageNeighborhood();
    boundary.releaseGeometry();
  }

  void setCellSet(SmartPointer<DenseCellSet<T, D>> passedCellSet) {
    cellSet = passedCellSet;
  }

  void setPointSource(const Vec3D<T> &passedOrigin,
                      const Vec3D<T> &passedDirection) {
    usePointSource = true;
    pointSourceOrigin = passedOrigin;
    pointSourceDirection = passedDirection;
  }

  template <typename ParticleType>
  void setParticle(std::unique_ptr<ParticleType> &p) {
    static_assert(std::is_base_of<AbstractParticle<T>, ParticleType>::value &&
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

  void setPrimaryDirection(const Vec3D<T> pPrimaryDirection) {
    primaryDirection = pPrimaryDirection;
    usePrimaryDirection = true;
  }

  void setExcludeMaterialId(int passedId) { excludeMaterialId = passedId; }

  SmartPointer<DenseCellSet<T, D>> getCellSet() const { return cellSet; }

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
    auto diskMesh = SmartPointer<lsMesh<T>>::New();
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

  inline Vec3D<T> calcMidPoint(const Vec3D<T> &minNode) {
    return Vec3D<T>{minNode[0] + mGridDelta / T(2),
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

} // namespace viennacs
