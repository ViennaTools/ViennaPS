#pragma once

#include <psDomain.hpp>
#include <psKDTree.hpp>
#include <psLogger.hpp>
#include <psUtils.hpp>

#include <rayGeometry.hpp>
#include <rayRNG.hpp>
#include <rayReflection.hpp>

template <class NumericType, int D> class psMeanFreePath {
public:
  psMeanFreePath() {}

  void setDomain(const psSmartPointer<psDomain<NumericType, D>> passedDomain) {
    domain = passedDomain;
    cellSet = domain->getCellSet();
    numCells = cellSet->getNumberOfCells();
    materialIds = cellSet->getScalarData("Material");
    cellSet->buildNeighborhood();
  }

  void setBulkLambda(const NumericType passedBulkLambda) {
    bulkLambda = passedBulkLambda;
  }

  void setMaterial(const psMaterial passedMaterial) {
    material = passedMaterial;
  }

  void setNumRaysPerCell(const NumericType passedNumRaysPerCell) {
    numRaysPerCell = passedNumRaysPerCell;
  }

  void setReflectionLimit(const int passedReflectionLimit) {
    reflectionLimit = passedReflectionLimit;
  }

  void setRngSeed(const unsigned int passedSeed) { seed = passedSeed; }

  void disableSmoothing() { smoothing = false; }

  void enableSmoothing() { smoothing = true; }

  void apply() {
    psLogger::getInstance().addInfo("Calculating mean free path ...").print();
    initGeometry();
    runKernel();
  }

private:
  void runKernel() {
    // thread local data storage
    const int numThreads = omp_get_max_threads();
    std::vector<std::vector<NumericType>> threadLocalData(numThreads);
    std::vector<std::vector<unsigned>> threadLocalHitCount(numThreads);

#pragma omp parallel
    {
      const int threadNum = omp_get_thread_num();
      auto &data = threadLocalData[threadNum];
      data.resize(numCells, 0.);
      auto &hitCount = threadLocalHitCount[threadNum];
      hitCount.resize(numCells, 0);
      std::uniform_int_distribution<unsigned> pointDist(0, numPoints - 1);

#pragma omp for schedule(dynamic)
      for (long long idx = 0; idx < numRays; ++idx) {

        if (threadNum == 0 && psLogger::getLogLevel() >= 4) {
          psUtils::printProgress(idx, numRays);
#ifdef VIENNAPS_PYTHON_BUILD
          if (PyErr_CheckSignals() != 0)
            throw pybind11::error_already_set();
#endif
        }

        // particle specific RNG seed
        auto particleSeed = rayInternal::tea<3>(idx, seed);
        rayRNG RngState(particleSeed);

        auto pointIdx = pointDist(RngState);
        auto direction = rayReflectionDiffuse<NumericType, D>(
            surfaceNormals[pointIdx], RngState);
        auto cellIdx = getStartingCell(surfacePoints[pointIdx]);
        auto origin = cellSet->getCellCenter(cellIdx);

        unsigned numReflections = 0;
        while (true) {

          /* -------- Cell Marching -------- */
          std::vector<int> hitCells(1, cellIdx);
          NumericType distance = 0;
          int prevIdx = -1;
          bool hitState = false; // -1 bulk hit, 1 material hit
          // invert direction for faster computation in intersectLineBox
          for (int i = 0; i < D; ++i) {
            direction[i] = 1. / direction[i];
          }

          while (true) {

            hitState = false;
            int currentCell = hitCells.back();
            int nextCell = -1;

            const auto &neighbors = cellSet->getNeighbors(currentCell);
            for (const auto &n : neighbors) {
              if (n < 0 || n == prevIdx) {
                continue;
              }

              if (!psMaterialMap::isMaterial(materialIds->at(n), material)) {
                hitState = true; // could be a hit
                continue;
              }

              auto &cellMin = cellSet->getNode(cellSet->getElement(n)[0]);
              auto &cellMax =
                  cellSet->getNode(cellSet->getElement(n)[D == 2 ? 2 : 6]);

              if (intersectLineBox(origin, direction, cellMin, cellMax,
                                   distance)) {
                nextCell = n;
                break;
              }
            }

            if (nextCell < 0 && hitState) {
              // hit a different material
              cellIdx = currentCell;
              break;
            }

            if (nextCell < 0) {
              // no hit
              distance = bulkLambda;
              break;
            }

            if (distance > bulkLambda) {
              // gas phase hit
              cellIdx = currentCell;
              break;
            }

            prevIdx = currentCell;
            hitCells.push_back(nextCell);
          }

          /* -------- Add to cells -------- */
          for (const auto &c : hitCells) {
            data[c] += distance + gridDelta;
            hitCount[c]++;
          }

          /* -------- Reflect -------- */
          if (!hitState)
            break;

          if (++numReflections >= reflectionLimit)
            break;

          // update origin
          origin = cellSet->getCellCenter(cellIdx);

          // update direction
          if (distance > bulkLambda) {
            // gas phase scatter
            randomDirection(direction, RngState);
          } else {
            // material reflection
            auto closestSurfacePoint = kdTree.findNearest(origin);
            assert(closestSurfacePoint->second < gridDelta);
            direction = rayReflectionDiffuse<NumericType, D>(
                surfaceNormals[closestSurfacePoint->first], RngState);
          }
        }
      }
    }

    // reduce data
    std::vector<NumericType> result(numCells, 0);
    for (const auto &data : threadLocalData) {
#pragma omp parallel for
      for (unsigned i = 0; i < numCells; ++i) {
        result[i] += data[i];
      }
    }

    // reduce hit counts
    std::vector<NumericType> hitCounts(numCells, 0);
    for (const auto &data : threadLocalHitCount) {
#pragma omp parallel for
      for (unsigned i = 0; i < numCells; ++i) {
        hitCounts[i] += data[i];
      }
    }

    // normalize data
#pragma omp parallel for
    for (unsigned i = 0; i < numCells; ++i) {
      if (hitCounts[i] > 0)
        result[i] = result[i] / hitCounts[i];
      else
        result[i] = 0.;
    }

    // smooth result
    auto finalResult = cellSet->addScalarData("MeanFreePath");
    materialIds = cellSet->getScalarData("Material");
#pragma omp parallel for
    for (unsigned i = 0; i < numCells; i++) {
      if (!psMaterialMap::isMaterial(materialIds->at(i), material))
        continue;

      if (smoothing) {
        const auto &neighbors = cellSet->getNeighbors(i);
        NumericType sum = 0;
        unsigned count = 0;
        for (const auto &n : neighbors) {
          if (n < 0 || !psMaterialMap::isMaterial(materialIds->at(n), material))
            continue;
          sum += result[n];
          count++;
        }
        if (count > 0)
          finalResult->at(i) = sum / count;
      } else {
        finalResult->at(i) = result[i];
      }
    }

    if (psLogger::getLogLevel() >= 4) {
      std::cout << std::endl;
    }
  }

  void initGeometry() {
    auto mesh = lsSmartPointer<lsMesh<NumericType>>::New();
    lsToDiskMesh<NumericType, D>(domain->getLevelSets()->back(), mesh).apply();
    surfacePoints = mesh->getNodes();
    surfaceNormals = *mesh->getCellData().getVectorData("Normals");
    numPoints = surfacePoints.size();

    kdTree.setPoints(surfacePoints);
    kdTree.build();

    gridDelta = domain->getGrid().getGridDelta();
    numRays = static_cast<long long>(numCells * numRaysPerCell);
  }

  int getStartingCell(const rayTriple<NumericType> &origin) const {
    int cellIdx = cellSet->getIndex(origin);
    if (cellIdx < 0) {
      psLogger::getInstance()
          .addError("No starting cell found for ray " +
                    std::to_string(origin[0]) + " " +
                    std::to_string(origin[1]) + " " + std::to_string(origin[2]))
          .print();
    }

    if (!psMaterialMap::isMaterial(materialIds->at(cellIdx), material)) {
      const auto &neighbors = cellSet->getNeighbors(cellIdx);
      for (const auto &n : neighbors) {
        if (n >= 0 && psMaterialMap::isMaterial(materialIds->at(n), material)) {
          cellIdx = n;
          break;
        }
      }
    }
    return cellIdx;
  }

  // https://gamedev.stackexchange.com/a/18459
  static bool intersectLineBox(const rayTriple<NumericType> &origin,
                               const rayTriple<NumericType> &direction,
                               const rayTriple<NumericType> &min,
                               const rayTriple<NumericType> &max,
                               NumericType &distance) {
    rayTriple<NumericType> t1, t2;
    for (int i = 0; i < D; ++i) {
      // direction is inverted
      t1[i] = (min[i] - origin[i]) * direction[i];
      t2[i] = (max[i] - origin[i]) * direction[i];
    }
    NumericType tmin, tmax;
    if constexpr (D == 2) {
      tmin = std::max(std::min(t1[0], t2[0]), std::min(t1[1], t2[1]));
      tmax = std::min(std::max(t1[0], t2[0]), std::max(t1[1], t2[1]));
    } else {
      tmin = std::max(std::max(std::min(t1[0], t2[0]), std::min(t1[1], t2[1])),
                      std::min(t1[2], t2[2]));
      tmax = std::min(std::min(std::max(t1[0], t2[0]), std::max(t1[1], t2[1])),
                      std::max(t1[2], t2[2]));
    }

    if (tmax > 0 && tmin < tmax) {
      // ray intersects box
      distance = tmin;
      return true;
    }

    return false;
  }

  static void randomDirection(rayTriple<NumericType> &direction,
                              rayRNG &rngState) {
    std::uniform_real_distribution<NumericType> dist(-1, 1);
    for (int i = 0; i < D; ++i) {
      direction[i] = dist(rngState);
    }
    rayInternal::Normalize(direction);
  }

private:
  psSmartPointer<psDomain<NumericType, D>> domain = nullptr;
  psSmartPointer<csDenseCellSet<NumericType, D>> cellSet = nullptr;
  psKDTree<NumericType, std::array<NumericType, 3>> kdTree;

  std::vector<NumericType> *materialIds;
  std::vector<std::array<NumericType, 3>> surfaceNormals;
  std::vector<std::array<NumericType, 3>> surfacePoints;

  NumericType bulkLambda = 0;
  NumericType gridDelta = 0;
  unsigned int numCells = 0;
  unsigned int numPoints = 0;
  unsigned int seed = 15235135;
  unsigned int reflectionLimit = 100;
  long long numRays = 0;
  NumericType numRaysPerCell = 1000;
  bool smoothing = true;
  psMaterial material = psMaterial::GAS;
};
