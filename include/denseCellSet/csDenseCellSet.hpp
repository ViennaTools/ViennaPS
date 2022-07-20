#ifndef DENSE_CELL_SET
#define DENSE_CELL_SET

#include <lsDomain.hpp>
#include <lsMakeGeometry.hpp>
#include <lsMesh.hpp>
#include <lsToVoxelMesh.hpp>
#include <lsVTKWriter.hpp>
#include <set>

#include "csBVH.hpp"
#include "csCellFiller.hpp"
#include "csTracePath.hpp"
#include "csUtil.hpp"

#define PERIODIC_BOUNDARY true

template <class T, int D> class csDenseCellSet {
private:
  using gridType = lsSmartPointer<lsMesh<>>;
  using levelSetsType =
      lsSmartPointer<std::vector<lsSmartPointer<lsDomain<T, D>>>>;

  gridType cellGrid = nullptr;
  lsSmartPointer<csCellFiller<T>> cellFiller = nullptr;
  levelSetsType levelSets = nullptr;
  lsSmartPointer<lsDomain<T, D>> surface = nullptr;
  lsSmartPointer<csBVH<T>> BVH = nullptr;
  std::vector<std::set<unsigned>> neighborhood;
  T gridDelta;
  size_t numberOfCells;
  T depth = 0.;
  int BVHlayers = 0;
  T meanFreePath = 0.;
  T meanFreePathStdDev = 0.;

public:
  csDenseCellSet() {}

  csDenseCellSet(levelSetsType passedLevelSets, T passedDepth = 0.)
      : levelSets(passedLevelSets) {
    fromLevelSets(passedLevelSets, passedDepth);
  }

  template <class CellFiller>
  void setCellFiller(lsSmartPointer<CellFiller> passedCellFiller) {
    cellFiller = std::dynamic_pointer_cast<csCellFiller<T>>(passedCellFiller);
  }

  void fromLevelSets(levelSetsType levelSets, T passedDepth = 0.) {

    if (cellGrid == nullptr)
      cellGrid = lsSmartPointer<lsMesh<>>::New();

    if (surface == nullptr)
      surface = lsSmartPointer<lsDomain<T, D>>::New(levelSets->back());
    else
      surface->deepCopy(levelSets->back());

    depth = passedDepth;

    lsToVoxelMesh<T, D> voxelConverter(cellGrid);
    if (depth > 0.) {
      auto plane = lsSmartPointer<lsDomain<T, D>>::New(surface->getGrid());
      T origin[D] = {0., 0., -depth};
      T normal[D] = {0., 0., 1.};
      lsMakeGeometry<T, D>(plane,
                           lsSmartPointer<lsPlane<T, D>>::New(origin, normal))
          .apply();
      voxelConverter.insertNextLevelSet(plane);
    }
    for (auto ls : *levelSets)
      voxelConverter.insertNextLevelSet(ls);
    voxelConverter.apply();

    gridDelta = surface->getGrid().getGridDelta();
    numberOfCells = cellGrid->getElements<(1 << D)>().size();

    std::vector<T> fillingFractions(numberOfCells, 0.);
    cellGrid->getCellData().insertNextScalarData(fillingFractions,
                                                 "fillingFraction");

    auto minBounds = surface->getGrid().getMinBounds();
    auto maxBounds = surface->getGrid().getMaxBounds();

    constexpr T eps = 1e-6;
    cellGrid->minimumExtent[0] = minBounds[0] * gridDelta - gridDelta - eps;
    cellGrid->minimumExtent[1] = minBounds[1] * gridDelta - gridDelta - eps;
    cellGrid->minimumExtent[2] = -depth - gridDelta - eps;

    cellGrid->maximumExtent[0] = maxBounds[0] * gridDelta + gridDelta + eps;
    cellGrid->maximumExtent[1] = maxBounds[1] * gridDelta + gridDelta + eps;
    cellGrid->maximumExtent[2] = maxBounds[2] * gridDelta + gridDelta + eps;

    auto minExtent = cellGrid->maximumExtent[0] - cellGrid->minimumExtent[0];
    minExtent = std::min(minExtent, cellGrid->maximumExtent[1] -
                                        cellGrid->minimumExtent[1]);
    minExtent = std::min(minExtent, cellGrid->maximumExtent[2] -
                                        cellGrid->minimumExtent[2]);

    BVHlayers = 0;
    while (minExtent / 2 > gridDelta) {
      BVHlayers++;
      minExtent /= 2;
    }
    BVH = lsSmartPointer<csBVH<T>>::New(getBoundingBox(), BVHlayers);
    buildNeighborhoodAndBVH();
  }

  csPair<csTriple<T>> getBoundingBox() const {
    return csPair<csTriple<T>>{cellGrid->minimumExtent,
                               cellGrid->maximumExtent};
  }

  void addScalarData(std::string name, T initValue) {
    std::vector<T> newData(numberOfCells, initValue);
    cellGrid->getCellData().insertNextScalarData(newData, name);
  }

  lsSmartPointer<csBVH<T>> getBVH() const { return BVH; }

  T getDepth() const { return depth; }

  T getGridDelta() const { return gridDelta; }

  gridType getCellGrid() const { return cellGrid; }

  levelSetsType getLevelSets() const { return levelSets; }

  size_t getNumberOfCells() const { return numberOfCells; }

  void cutSurface(lsSmartPointer<lsDomain<T, D>> advectedSurface) {
    auto cutCellGrid = lsSmartPointer<lsMesh<>>::New();

    lsToVoxelMesh<T, D> voxelConverter(cutCellGrid);
    if (depth > 0.) {
      auto plane = lsSmartPointer<lsDomain<T, D>>::New(surface->getGrid());
      T origin[D] = {0., 0., -depth};
      T normal[D] = {0., 0., 1.};
      lsMakeGeometry<T, D>(plane,
                           lsSmartPointer<lsPlane<T, D>>::New(origin, normal))
          .apply();
      voxelConverter.insertNextLevelSet(plane);
    }
    voxelConverter.insertNextLevelSet(advectedSurface);
    voxelConverter.insertNextLevelSet(surface);
    voxelConverter.apply();

    auto cutMatIds = cutCellGrid->getCellData().getScalarData("Material");
    auto &hexas = cellGrid->getElements<(1 << D)>();

    const auto nCutCells = cutCellGrid->getElements<(1 << D)>().size();

    size_t offset = 0;
    if (numberOfCells > nCutCells)
      offset = numberOfCells - nCutCells;

    auto numScalarData = cellGrid->getCellData().getScalarDataSize();

    for (int elIdx = nCutCells - 1; elIdx >= 0; elIdx--) {
      if (cutMatIds->at(elIdx) == 2) {
        for (int i = 0; i < numScalarData; i++) {
          auto data = cellGrid->getCellData().getScalarData(i);
          data->erase(data->begin() + elIdx + offset);
        }
        hexas.erase(hexas.begin() + elIdx + offset);
      }
    }
    numberOfCells = hexas.size();
    surface->deepCopy(advectedSurface);
    buildNeighborhoodAndBVH();
  }

  //   void
  //   updateNeighborhood()
  //   {
  //     // update neighborhood relations(still quite slow)
  //     // maybe better(if possible) to just rebuild them
  //     for (const auto removedIdx : removedInidices)
  //       neighborhood.erase(neighborhood.begin() + removedIdx);
  // #pragma omp parallel for
  //     for (size_t nIdx = 0; nIdx < neighborhood.size(); nIdx++)
  //     {
  //       for (const auto removedIdx : removedInidices)
  //       {
  //         if (removedIdx > *neighborhood[nIdx].rbegin())
  //           continue;
  //         else
  //           neighborhood[nIdx].erase(removedIdx);
  //       }
  //       std::vector<int> subtractArray(neighborhood[nIdx].size(), 0);
  //       for (const auto removedIdx : removedInidices)
  //       {
  //         if (removedIdx > *neighborhood[nIdx].rbegin())
  //           continue;
  //         else
  //         {
  //           const auto it = neighborhood[nIdx].lower_bound(removedIdx);
  //           const auto dist = std::distance(neighborhood[nIdx].begin(),
  //                                           it);
  //           for (size_t i = dist; i < subtractArray.size(); i++)
  //             ++subtractArray[i];
  //         }
  //       }
  //       size_t subIdx = 0;
  //       for (auto it{neighborhood[nIdx].begin()},
  //            end{neighborhood[nIdx].end()};
  //            it != end;)
  //       {
  //         if (subtractArray[subIdx] > 0)
  //         {
  //           subtractArray[subIdx] = *it - subtractArray[subIdx];
  //           it = neighborhood[nIdx].erase(it);
  //         }
  //         else
  //         {
  //           subtractArray[subIdx] = -1;
  //           ++it;
  //         }
  //         ++subIdx;
  //       }
  //       for (auto el : subtractArray)
  //       {
  //         if (el >= 0)
  //           neighborhood[nIdx].insert(el);
  //       }
  //     }
  //   }

  void printNeighborhood() {
    for (size_t i = 0; i < numberOfCells; i++) {
      std::cout << i << ": ";
      for (const auto n : neighborhood[i]) {
        std::cout << n << ", ";
      }
      std::cout << std::endl;
    }
  }

  bool setFillingFraction(int idx, T fill) {
    if (idx < 0)
      return false;

    getFillingFractions()->at(idx) = fill;
    return true;
  }

  bool setFillingFraction(T x, T y, T z, T fill) {
    auto idx = findIndex(csTriple<T>{x, y, z});
    return setFillingFraction(idx, fill);
  }

  bool setFillingFraction(const csTriple<T> point, T fill) {
    auto idx = findIndex(point);
    return setFillingFraction(idx, fill);
  }

  void setMeanFreePath(const T passedMeanFreePath, const T stdDev) {
    meanFreePath = passedMeanFreePath;
    meanFreePathStdDev = stdDev;
  }

  void writeVTU(std::string fileName) {
    lsVTKWriter<T>(cellGrid, fileName).apply();
  }

  void clear() {
    auto ff = getFillingFractions();
    std::fill(ff->begin(), ff->end(), 0.);
  }

  void traceOnArea(csTracePath<T> &path, const csTriple<T> &hitPoint,
                   csTriple<T> direction, const T fillStart) {
    auto &cells = cellGrid->getElements<(1 << D)>();
    auto &nodes = cellGrid->getNodes();
    auto materialIds = cellGrid->getCellData().getScalarData("Material");

    const auto xExt = cellGrid->maximumExtent[0] - cellGrid->minimumExtent[0];
    const auto yExt = cellGrid->maximumExtent[1] - cellGrid->minimumExtent[1];
    const auto gd2 = gridDelta / 2.;

    // normalize(direction);
#ifdef ARCH_X86
    __m128 SSEdirection =
        _mm_set_ps(0.f, direction[2], direction[1], direction[0]);
    SSEdirection = NormalizeAccurateSse(SSEdirection);
    __m128 SSEhitpoint = _mm_set_ps(0.f, hitPoint[2], hitPoint[1], hitPoint[0]);

    for (size_t idx = 0; idx < numberOfCells; idx++) {
      T dist = std::numeric_limits<T>::max();
      T normalDist = std::numeric_limits<T>::max();

      if constexpr (PERIODIC_BOUNDARY) {
        for (int i = -1; i < 2; i++) {
          for (int j = -1; j < 2; j++) {
            float tmpDist, tmpNormalDist;

            __m128 SSEpoint =
                _mm_set_ps(0.f, nodes[cells[idx][0]][2] + gd2,
                           nodes[cells[idx][0]][1] + j * yExt + gd2,
                           nodes[cells[idx][0]][0] + i * xExt + gd2);

            SSEpoint = _mm_sub_ps(SSEpoint, SSEhitpoint);
            tmpNormalDist = DotProductSse(SSEpoint, SSEdirection);
            tmpDist = NormSse(CrossProductSse(SSEpoint, SSEdirection));

            if (tmpDist < dist) {
              dist = tmpDist;
              normalDist = tmpNormalDist;
            }
          }
        }
      } else {
        const auto point = calcMidPoint(nodes[cells[idx][0]]);
        pointLineDistance(normalDist, dist, point, hitPoint, direction);
      }

      auto fill = cellFiller->fillArea(idx, normalDist, dist, fillStart,
                                       materialIds->at(idx));
      path.addGridData(idx, fill);
    }
#else
    std::cerr << "Not implemented in ARM architecture." << std::endl;
#endif
  }

  void traceOnPath(csTracePath<T> &path, csTriple<T> hitPoint,
                   csTriple<T> direction, const T fillStart,
                   const T stepDistance, rayRNG &RNG) {
    scaleToLength(direction, stepDistance);
    T distance = 0.;
    T fill = 0.;
    T energy = fillStart;

    // find surface hitpoint
    auto prevIdx = findIndex(hitPoint);
    size_t sanityCounter = 0;
    while (prevIdx < 0) {
      add(hitPoint, direction);
      if (++sanityCounter > 10 || !checkBoundsPeriodic(hitPoint)) {
        return;
      }
      prevIdx = findIndex(hitPoint);
    }

    auto materialId =
        cellGrid->getCellData().getScalarData("Material")->at(prevIdx);
    fill = cellFiller->fill(prevIdx, distance, energy, materialId, hitPoint,
                            direction, stepDistance, RNG);
    path.addPoint(prevIdx, fill);
    add(hitPoint, direction);
    distance += stepDistance;

    while (checkBoundsPeriodic(hitPoint)) {
      auto newIdx = findIndexNearPrevious(hitPoint, prevIdx);
      if (newIdx != prevIdx && newIdx >= 0) {
        materialId =
            cellGrid->getCellData().getScalarData("Material")->at(newIdx);
        fill = cellFiller->fill(newIdx, distance, energy, materialId, hitPoint,
                                direction, stepDistance, RNG);
        path.addPoint(newIdx, fill);
        if (energy < 0)
          break;
        prevIdx = newIdx;
      }
      add(hitPoint, direction);
      distance += stepDistance;
    }
  }

  void traceOnPathCascade(csTracePath<T> &path, csTriple<T> hitPoint,
                          csTriple<T> direction, const T startEnergy,
                          const T stepDistance, rayRNG &RNG) {

    scaleToLength(direction, stepDistance);
    T fill = 0.;
    std::vector<Particle<T>> particleStack;

    // find surface hitpoint
    auto prevIdx = findIndex(hitPoint);
    size_t sanityCounter = 0;
    while (prevIdx < 0) {
      add(hitPoint, direction);
      if (++sanityCounter > 10 || !checkBoundsPeriodic(hitPoint)) {
        return;
      }
      prevIdx = findIndex(hitPoint);
    }

    particleStack.emplace_back(
        Particle<T>{hitPoint, direction, startEnergy, 0., prevIdx, 0});

    while (!particleStack.empty()) {
      auto particle = std::move(particleStack.back());
      particleStack.pop_back();

      // trace particle
      while (particle.energy >= 0 && checkBoundsPeriodic(particle.position)) {
        auto newIdx = findIndexNearPrevious(particle.position, particle.cellId);

        if (newIdx != particle.cellId && newIdx >= 0) {
          particle.cellId = newIdx;
          fill =
              cellFiller->cascade(particle, stepDistance, RNG, particleStack);
          path.addGridData(newIdx, fill);
          particle.distance = 0.; // reset particle distance
        }
        add(particle.position, particle.direction);
        particle.distance += stepDistance;
      }
    }
  }

  void traceCollisionPath(csTracePath<T> &path, csTriple<T> hitPoint,
                          csTriple<T> direction, const T startEnergy,
                          rayRNG &RNG) {
    T fill = 0.;
    T dist;
    std::vector<Particle<T>> particleStack;

    std::normal_distribution<T> normalDist{meanFreePath, meanFreePathStdDev};

    particleStack.emplace_back(
        Particle<T>{hitPoint, direction, startEnergy, 0., -1, 0});

    while (!particleStack.empty()) {
      auto particle = std::move(particleStack.back());
      particleStack.pop_back();

      // trace particle
      while (particle.energy >= 0) {
        particle.distance = -1;
        while (particle.distance < 0)
          particle.distance = normalDist(RNG);
        auto travelDist = multNew(particle.direction, particle.distance);
        add(particle.position, travelDist);

        if (!checkBoundsPeriodic(particle.position))
          break;

        auto newIdx = findIndex(particle.position);
        if (newIdx < 0)
          break;

        if (newIdx != particle.cellId) {
          particle.cellId = newIdx;
          fill = cellFiller->collision(particle, RNG, particleStack);
          path.addGridData(newIdx, fill);
        }
      }
    }
  }

  void mergePath(csTracePath<T> &path, T factor = 1.) {
    auto ff = getFillingFractions();
    if (!path.getData().empty()) {
      for (const auto it : path.getData()) {
        ff->at(it.first) += it.second / factor;
      }
    }

    if (!path.getGridData().empty()) {
      const auto &data = path.getGridData();
      for (size_t idx = 0; idx < numberOfCells; idx++) {
        ff->at(idx) += data[idx] / factor;
      }
    }
  }

  std::set<unsigned> &getNeighbors(size_t idx) { return neighborhood[idx]; }

  std::vector<T> *getFillingFractions() const {
    return cellGrid->getCellData().getScalarData("fillingFraction");
  }

  T getFillingFraction(const csTriple<T> &point) {
    auto idx = findIndex(point);
    if (idx < 0)
      return -1.;

    return getFillingFractions()->at(idx);
  }

  std::vector<T> *getScalarData(std::string name) {
    return cellGrid->getCellData().getScalarData(name);
  }

  lsSmartPointer<lsDomain<T, D>> getSurface() { return surface; }

private:
  int findSurfaceHitPoint(csTriple<T> &hitPoint, const csTriple<T> &direction) {
    // find surface hitpoint
    auto idx = findIndex(hitPoint);

    if (idx > 0)
      return idx;

    auto moveDirection = multNew(direction, gridDelta / 5.);
    size_t sanityCounter = 0;
    while (idx < 0) {
      add(hitPoint, moveDirection);
      if (++sanityCounter > 100 || !checkBoundsPeriodic(hitPoint)) {
        return -1;
      }
      idx = findIndex(hitPoint);
    }

    return idx;
  }

  void pointLineDistance(T &normalDist, T &dist, csTriple<T> &point,
                         const csTriple<T> &lineStart,
                         const csTriple<T> &lineDirection) {
    sub(point, lineStart);
    normalDist = dot(point, lineDirection);
    dist = norm(crossProd(point, lineDirection));
    // // projected distance on line
    // multAdd(tmp, lineDirection, lineStart,
    //         normalDist); // tmp = dir * normDist + hit
    // dist = distance(tmp, point);
  }

  bool checkBounds(const csTriple<T> &hitPoint) const {
    const auto &min = cellGrid->minimumExtent;
    const auto &max = cellGrid->maximumExtent;

    return hitPoint[0] >= min[0] && hitPoint[0] <= max[0] &&
           hitPoint[1] >= min[1] && hitPoint[1] <= max[1] &&
           hitPoint[2] >= min[2] && hitPoint[2] <= max[2];
  }

  bool checkBoundsPeriodic(csTriple<T> &hitPoint) const {
    const auto &min = cellGrid->minimumExtent;
    const auto &max = cellGrid->maximumExtent;

    if (hitPoint[2] < min[2] || hitPoint[2] > max[2])
      return false;

    if (hitPoint[0] < min[0]) {
      hitPoint[0] = max[0] - gridDelta / 2.;
    } else if (hitPoint[0] > max[0]) {
      hitPoint[0] = min[0] + gridDelta / 2.;
    }

    if (hitPoint[1] < min[1]) {
      hitPoint[1] = max[1] - gridDelta / 2.;
    } else if (hitPoint[1] > max[1]) {
      hitPoint[1] = min[1] + gridDelta / 2.;
    }

    return true;
  }

  void scaleToLength(csTriple<T> &vec, T length) {
    const auto vecLength =
        std::sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);

    for (size_t i = 0; i < D; i++)
      vec[i] *= length / vecLength;
  }

  void add(std::array<T, D> &vec, const std::array<T, D> &vec2) {
    for (size_t i = 0; i < D; i++)
      vec[i] += vec2[i];
  }

  int findIndexNearPrevious(const csTriple<T> &point, int prevIdx) {
    auto &hexas = cellGrid->getElements<(1 << D)>();
    auto &nodes = cellGrid->getNodes();
    auto ff = getFillingFractions();

    int idx = -1;

    // search in neighborhood of previous index
    for (const auto &i : neighborhood[prevIdx]) {
      if (isInsideHexa(point, nodes[hexas[i][0]])) {
        idx = i;
        break;
      }
    }

    if (idx != -1)
      return idx;

    return findIndex(point);
  }

  int findIndexTriangles(const std::array<T, 2> &point) {
    auto &triangles = cellGrid->getElements<3>();
    auto &nodes = cellGrid->getNodes();
    int idx = -1;
    const auto numPoints = triangles.size();

    for (int cellId = 0; cellId < numPoints; cellId++) {
      if (isInsideTriangle(point, nodes[triangles[cellId][0]],
                           nodes[triangles[cellId][1]],
                           nodes[triangles[cellId][2]])) {
        idx = cellId;
        break;
      }
    }
    return idx;
  }

  int findIndex(const csTriple<T> &point) {
    auto &elems = cellGrid->getElements<(1 << D)>();
    auto &nodes = cellGrid->getNodes();
    int idx = -1;

    auto cellIds = BVH->getCellIds(point);
    for (const auto cellId : *cellIds) {
      if (isInsideHexa(point, nodes[elems[cellId][0]])) {
        idx = cellId;
        break;
      }
    }
    return idx;
  }

  bool isInsideTetra(const std::array<T, 2> &point,
                     const csTriple<T> &tetraMin) {
    return point[0] >= tetraMin[0] && point[0] <= (tetraMin[0] + gridDelta) &&
           point[1] >= tetraMin[1] && point[1] <= (tetraMin[1] + gridDelta) &&
  }

  bool isInsideTriangle(const std::array<T, 2> &point,
                        const std::array<T, 3> &p0, const std::array<T, 3> &p1,
                        const std::array<T, 3> &p2) {
    auto s = (p0[1] * p2[0] - p0[0] * p2[1] + (p2[1] - p0[1]) * point[0] +
              (p0[0] - p2[0]) * point[1]);
    auto t = (p0[0] * p1[1] - p0[1] * p1[0] + (p0[1] - p1[1]) * point[0] +
              (p1[0] - p0[0]) * point[1]);
    return s > 0 && t > 0 && (1 - s - t) > 0;
  }

  bool isInsideHexa(const csTriple<T> &point, const csTriple<T> &hexaMin) {
    return point[0] >= hexaMin[0] && point[0] <= (hexaMin[0] + gridDelta) &&
           point[1] >= hexaMin[1] && point[1] <= (hexaMin[1] + gridDelta) &&
           point[2] >= hexaMin[2] && point[2] <= (hexaMin[2] + gridDelta);
  }

  void buildNeighborhoodAndBVH() {
    auto &hexas = cellGrid->getElements<(1 << D)>();
    auto &nodes = cellGrid->getNodes();

    std::vector<std::vector<unsigned>> nodeHexaConnections(nodes.size());
    neighborhood.clear();
    neighborhood.resize(hexas.size());

    BVH->clearCellIds();

    for (size_t hexaIdx = 0; hexaIdx < hexas.size(); hexaIdx++) {
      for (size_t n = 0; n < 8; n++) {
        nodeHexaConnections[hexas[hexaIdx][n]].push_back(hexaIdx);
        auto &node = nodes[hexas[hexaIdx][n]];
        BVH->getCellIds(node)->insert(hexaIdx);
      }
    }

    for (size_t nodeIdx = 0; nodeIdx < nodes.size(); nodeIdx++) {
      for (size_t hexInsertIdx = 0;
           hexInsertIdx < nodeHexaConnections[nodeIdx].size(); hexInsertIdx++) {
        for (size_t hexIdx = 0; hexIdx < nodeHexaConnections[nodeIdx].size();
             hexIdx++) {
          neighborhood[nodeHexaConnections[nodeIdx][hexInsertIdx]].insert(
              nodeHexaConnections[nodeIdx][hexIdx]);
        }
      }
    }
  }

  inline csTriple<T> calcMidPoint(const csTriple<T> &minNode) {
    return csTriple<T>{minNode[0] + gridDelta / 2., minNode[1] + gridDelta / 2.,
                       minNode[2] + gridDelta / 2.};
  }
};

#endif