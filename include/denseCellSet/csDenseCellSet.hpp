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

#define CS_PERIODIC_BOUNDARY true

template <class T, int D> class csDenseCellSet {
private:
  using gridType = lsSmartPointer<lsMesh<>>;
  using levelSetsType =
      lsSmartPointer<std::vector<lsSmartPointer<lsDomain<T, D>>>>;

  gridType cellGrid = nullptr;
  lsSmartPointer<csCellFiller<T>> cellFiller = nullptr;
  levelSetsType levelSets = nullptr;
  lsSmartPointer<lsDomain<T, D>> surface = nullptr;
  lsSmartPointer<csBVH<T, D>> BVH = nullptr;
  std::vector<std::set<unsigned>> neighborhood;
  T gridDelta;
  size_t numberOfCells;
  T depth = 0.;
  int BVHlayers = 0;
  T meanFreePath = 0.;
  T meanFreePathStdDev = 0.;
  bool cellSetAboveSurface = false;

public:
  csDenseCellSet() {}

  csDenseCellSet(levelSetsType passedLevelSets, T passedDepth = 0.,
                 bool passedCellSetPosition = false)
      : levelSets(passedLevelSets), cellSetAboveSurface(passedCellSetPosition) {
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
    auto plane = lsSmartPointer<lsDomain<T, D>>::New(surface->getGrid());
    if (depth != 0.) {
      T origin[D] = {0.};
      T normal[D] = {0.};
      origin[D - 1] = depth;
      normal[D - 1] = 1.;
      lsMakeGeometry<T, D>(plane,
                           lsSmartPointer<lsPlane<T, D>>::New(origin, normal))
          .apply();
    }
    if (!cellSetAboveSurface && depth != 0)
      voxelConverter.insertNextLevelSet(plane);
    for (auto ls : *levelSets)
      voxelConverter.insertNextLevelSet(ls);
    if (cellSetAboveSurface && depth != 0) {
      voxelConverter.insertNextLevelSet(plane);
    }
    voxelConverter.apply();

    gridDelta = surface->getGrid().getGridDelta();
    numberOfCells = cellGrid->getElements<(1 << D)>().size();

    std::vector<T> fillingFractions(numberOfCells, 0.);
    cellGrid->getCellData().insertNextScalarData(fillingFractions,
                                                 "fillingFraction");

    auto minBounds = surface->getGrid().getMinBounds();
    auto maxBounds = surface->getGrid().getMaxBounds();

    constexpr T eps = 1e-6;

    if constexpr (D == 3) {
      cellGrid->minimumExtent[0] = minBounds[0] * gridDelta - gridDelta - eps;
      cellGrid->minimumExtent[1] = minBounds[1] * gridDelta - gridDelta - eps;

      cellGrid->maximumExtent[0] = maxBounds[0] * gridDelta + gridDelta + eps;
      cellGrid->maximumExtent[1] = maxBounds[1] * gridDelta + gridDelta + eps;

      if (!cellSetAboveSurface && depth < 0) {
        cellGrid->minimumExtent[2] = depth - gridDelta - eps;
        cellGrid->maximumExtent[2] = maxBounds[2] * gridDelta + gridDelta + eps;
      } else if (cellSetAboveSurface && depth > 0) {
        cellGrid->minimumExtent[2] = minBounds[2] * gridDelta - gridDelta - eps;
        cellGrid->maximumExtent[2] = depth + gridDelta + eps;
      } else {
        cellGrid->minimumExtent[2] = minBounds[2] * gridDelta - gridDelta - eps;
        cellGrid->maximumExtent[2] = maxBounds[2] * gridDelta + gridDelta + eps;
      }
    } else {
      cellGrid->minimumExtent[0] = minBounds[0] * gridDelta - gridDelta - eps;
      cellGrid->maximumExtent[0] = maxBounds[0] * gridDelta + gridDelta + eps;

      if (!cellSetAboveSurface && depth < 0) {
        cellGrid->minimumExtent[1] = depth - gridDelta - eps;
        cellGrid->maximumExtent[1] = maxBounds[1] * gridDelta + gridDelta + eps;
      } else if (cellSetAboveSurface && depth > 0) {
        cellGrid->minimumExtent[1] = minBounds[1] * gridDelta - gridDelta - eps;
        cellGrid->maximumExtent[1] = depth + gridDelta + eps;
      } else {
        cellGrid->minimumExtent[1] = minBounds[1] * gridDelta - gridDelta - eps;
        cellGrid->maximumExtent[1] = maxBounds[1] * gridDelta + gridDelta + eps;
      }
    }

    auto minExtent = cellGrid->maximumExtent[0] - cellGrid->minimumExtent[0];
    minExtent = std::min(minExtent, cellGrid->maximumExtent[1] -
                                        cellGrid->minimumExtent[1]);
    if constexpr (D == 3)
      minExtent = std::min(minExtent, cellGrid->maximumExtent[2] -
                                          cellGrid->minimumExtent[2]);

    BVHlayers = 0;
    while (minExtent / 2 > gridDelta) {
      BVHlayers++;
      minExtent /= 2;
    }
    BVH = lsSmartPointer<csBVH<T, D>>::New(getBoundingBox(), BVHlayers);
    buildNeighborhoodAndBVH();
  }

  csPair<std::array<T, D>> getBoundingBox() const {
    if constexpr (D == 3)
      return csPair<csTriple<T>>{cellGrid->minimumExtent,
                                 cellGrid->maximumExtent};
    else
      return csPair<csPair<T>>{
          cellGrid->minimumExtent[0], cellGrid->minimumExtent[1],
          cellGrid->maximumExtent[0], cellGrid->maximumExtent[1]};
  }

  void addScalarData(std::string name, T initValue) {
    std::vector<T> newData(numberOfCells, initValue);
    cellGrid->getCellData().insertNextScalarData(newData, name);
  }

  lsSmartPointer<csBVH<T, D>> getBVH() const { return BVH; }

  T getDepth() const { return depth; }

  T getGridDelta() const { return gridDelta; }

  gridType getCellGrid() const { return cellGrid; }

  levelSetsType getLevelSets() const { return levelSets; }

  size_t getNumberOfCells() const { return numberOfCells; }

  void updateMaterials() {
    auto numScalarData = cellGrid->getCellData().getScalarDataSize();
    std::vector<std::vector<T>> scalarData(numScalarData - 1);
    std::vector<std::string> scalarDataLabels(numScalarData - 1);

    int n = 0;
    for (int i = 0; i < numScalarData; i++) {
      auto label = cellGrid->getCellData().getScalarDataLabel(i);
      if (label == "Material")
        continue;
      auto data = cellGrid->getCellData().getScalarData(i);
      scalarData[n] = std::move(*data);
      scalarDataLabels[n++] = label;
    }

    lsToVoxelMesh<T, D> voxelConverter(cellGrid);
    auto plane =
        lsSmartPointer<lsDomain<T, D>>::New(levelSets->back()->getGrid());
    if (depth != 0.) {
      T origin[D] = {0.};
      T normal[D] = {0.};
      origin[D - 1] = depth;
      normal[D - 1] = 1.;
      lsMakeGeometry<T, D>(plane,
                           lsSmartPointer<lsPlane<T, D>>::New(origin, normal))
          .apply();
    }
    if (!cellSetAboveSurface && depth != 0)
      voxelConverter.insertNextLevelSet(plane);
    for (auto ls : *levelSets)
      voxelConverter.insertNextLevelSet(ls);
    if (cellSetAboveSurface && depth != 0) {
      voxelConverter.insertNextLevelSet(plane);
    }
    voxelConverter.apply();

    if (numberOfCells != cellGrid->getElements<(1 << D)>().size()) {
      std::cerr << "Removing cells when not allowed." << std::endl;
    }

    for (int i = 0; i < numScalarData - 1; i++) {
      cellGrid->getCellData().insertNextScalarData(scalarData[i],
                                                   scalarDataLabels[i]);
    }
  }

  void cutSurface(lsSmartPointer<lsDomain<T, D>> advectedSurface) {
    auto cutCellGrid = lsSmartPointer<lsMesh<>>::New();

    lsToVoxelMesh<T, D> voxelConverter(cutCellGrid);
    if (depth > 0.) {
      auto plane = lsSmartPointer<lsDomain<T, D>>::New(surface->getGrid());
      T origin[D] = {0., 0., depth};
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

  bool setFillingFraction(const std::array<T, 3> &point, T fill) {
    auto idx = findIndex(point);
    return setFillingFraction(idx, fill);
  }

  bool addFillingFraction(int idx, T fill) {
    if (idx < 0)
      return false;

    getFillingFractions()->at(idx) += fill;
    return true;
  }

  bool addFillingFraction(const std::array<T, 3> &point, T fill) {
    auto idx = findIndex(point);
    return addFillingFraction(idx, fill);
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

  void traceArea(csTracePath<T> &path, const csTriple<T> &hitPoint,
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

      if constexpr (CS_PERIODIC_BOUNDARY) {
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
    std::cerr << "Only implemented in x86 architecture." << std::endl;
#endif
  }

  void traceCascadePath(csTracePath<T> &path, csTriple<T> hitPoint,
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

  void traceCascadeCollision(csTracePath<T> &path, csTriple<T> hitPoint,
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

  T getFillingFraction(const std::array<T, D> &point) {
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
           point[1] >= tetraMin[1] && point[1] <= (tetraMin[1] + gridDelta);
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
    auto &elems = cellGrid->getElements<(1 << D)>();
    auto &nodes = cellGrid->getNodes();

    std::vector<std::vector<unsigned>> nodeElemConnections(nodes.size());
    neighborhood.clear();
    neighborhood.resize(elems.size());
    BVH->clearCellIds();

    for (size_t elemIdx = 0; elemIdx < elems.size(); elemIdx++) {
      for (size_t n = 0; n < (1 << D); n++) {
        nodeElemConnections[elems[elemIdx][n]].push_back(elemIdx);
        auto &node = nodes[elems[elemIdx][n]];
        BVH->getCellIds(node)->insert(elemIdx);
      }
    }

    for (size_t nodeIdx = 0; nodeIdx < nodes.size(); nodeIdx++) {
      for (size_t elemInsertIdx = 0;
           elemInsertIdx < nodeElemConnections[nodeIdx].size();
           elemInsertIdx++) {
        for (size_t elemIdx = 0; elemIdx < nodeElemConnections[nodeIdx].size();
             elemIdx++) {
          neighborhood[nodeElemConnections[nodeIdx][elemInsertIdx]].insert(
              nodeElemConnections[nodeIdx][elemIdx]);
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