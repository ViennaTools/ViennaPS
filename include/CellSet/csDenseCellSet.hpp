#ifndef DENSE_CELL_SET
#define DENSE_CELL_SET

#include <csBVH.hpp>
#include <csTracePath.hpp>
#include <csUtil.hpp>

#include <lsDomain.hpp>
#include <lsMakeGeometry.hpp>
#include <lsMesh.hpp>
#include <lsToSurfaceMesh.hpp>
#include <lsToVoxelMesh.hpp>
#include <lsVTKWriter.hpp>

#include <rayUtil.hpp>

#include <psLogger.hpp>
#include <psSmartPointer.hpp>

/**
  This class represents a cell-based voxel implementation of a volume. The
  depth of the cell set in z-direction can be specified.
*/
template <class T, int D> class csDenseCellSet {
private:
  using gridType = psSmartPointer<lsMesh<T>>;
  using levelSetsType =
      psSmartPointer<std::vector<psSmartPointer<lsDomain<T, D>>>>;

  levelSetsType levelSets = nullptr;
  gridType cellGrid = nullptr;
  psSmartPointer<lsDomain<T, D>> surface = nullptr;
  psSmartPointer<csBVH<T, D>> BVH = nullptr;
  std::vector<std::array<int, 2 * D>> cellNeighbors; // -x, x, -y, y, -z, z
  T gridDelta;
  size_t numberOfCells;
  T depth = 0.;
  int BVHlayers = 0;
  bool cellSetAboveSurface = false;
  std::vector<T> *fillingFractions;
  const T eps = 1e-4;
  hrleVectorType<hrleIndexType, D> minIndex, maxIndex;

public:
  csDenseCellSet() {}

  csDenseCellSet(levelSetsType passedLevelSets, T passedDepth = 0.,
                 bool passedCellSetPosition = false)
      : levelSets(passedLevelSets), cellSetAboveSurface(passedCellSetPosition) {
    fromLevelSets(passedLevelSets, passedDepth);
  }

  void fromLevelSets(levelSetsType passedLevelSets, T passedDepth = 0.) {
    levelSets = passedLevelSets;

    if (cellGrid == nullptr)
      cellGrid = psSmartPointer<lsMesh<T>>::New();

    if (surface == nullptr)
      surface = psSmartPointer<lsDomain<T, D>>::New(levelSets->back());
    else
      surface->deepCopy(levelSets->back());

    gridDelta = surface->getGrid().getGridDelta();

    depth = passedDepth;
    std::vector<psSmartPointer<lsDomain<T, D>>> levelSetsInOrder;
    auto plane = psSmartPointer<lsDomain<T, D>>::New(surface->getGrid());
    {
      T origin[D] = {0.};
      T normal[D] = {0.};
      origin[D - 1] = depth;
      normal[D - 1] = 1.;
      lsMakeGeometry<T, D>(plane,
                           psSmartPointer<lsPlane<T, D>>::New(origin, normal))
          .apply();
    }
    if (!cellSetAboveSurface)
      levelSetsInOrder.push_back(plane);
    for (auto ls : *levelSets)
      levelSetsInOrder.push_back(ls);
    if (cellSetAboveSurface)
      levelSetsInOrder.push_back(plane);

    calculateMinMaxIndex(levelSetsInOrder);
    lsToVoxelMesh<T, D>(levelSetsInOrder, cellGrid).apply();
    // lsToVoxelMesh also saves the extent in the cell grid

#ifndef NDEBUG
    int db_ls = 0;
    for (auto &ls : levelSetsInOrder) {
      auto mesh = psSmartPointer<lsMesh<T>>::New();
      lsToSurfaceMesh<T, D>(ls, mesh).apply();
      lsVTKWriter<T>(mesh, "cellSet_debug_" + std::to_string(db_ls++) + ".vtp")
          .apply();
    }
    lsVTKWriter<T>(cellGrid, "cellSet_debug_init.vtu").apply();
#endif

    if (!cellSetAboveSurface)
      adjustMaterialIds();

    // create filling fractions as default scalar cell data
    numberOfCells = cellGrid->template getElements<(1 << D)>().size();
    std::vector<T> fillingFractionsTemp(numberOfCells, 0.);

    cellGrid->getCellData().insertNextScalarData(
        std::move(fillingFractionsTemp), "fillingFraction");
    fillingFractions = cellGrid->getCellData().getScalarData("fillingFraction");

    // calculate number of BVH layers
    for (unsigned i = 0; i < D; ++i) {
      cellGrid->minimumExtent[i] -= eps;
      cellGrid->maximumExtent[i] += eps;
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

    BVH = psSmartPointer<csBVH<T, D>>::New(getBoundingBox(), BVHlayers);
    buildBVH();
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

  std::vector<T> *addScalarData(std::string name, T initValue) {
    std::vector<T> newData(numberOfCells, initValue);
    cellGrid->getCellData().insertNextScalarData(std::move(newData), name);
    fillingFractions = cellGrid->getCellData().getScalarData("fillingFraction");
    return cellGrid->getCellData().getScalarData(name);
  }

  gridType getCellGrid() { return cellGrid; }

  T getDepth() const { return depth; }

  T getGridDelta() const { return gridDelta; }

  std::vector<std::array<T, 3>> &getNodes() const {
    return cellGrid->getNodes();
  }

  std::vector<std::array<unsigned, (1 << D)>> getElements() {
    return cellGrid->template getElements<(1 << D)>();
  }

  psSmartPointer<lsDomain<T, D>> getSurface() { return surface; }

  levelSetsType getLevelSets() const { return levelSets; }

  size_t getNumberOfCells() const { return numberOfCells; }

  std::vector<T> *getFillingFractions() const { return fillingFractions; }

  T getFillingFraction(const std::array<T, D> &point) {
    auto idx = findIndex(point);
    if (idx < 0)
      return -1.;

    return getFillingFractions()->at(idx);
  }

  int getIndex(std::array<T, 3> &point) { return findIndex(point); }

  std::vector<T> *getScalarData(std::string name) {
    return cellGrid->getCellData().getScalarData(name);
  }

  // Set whether the cell set should be created below (false) or above (true)
  // the surface.
  void setCellSetPosition(const bool passedCellSetPosition) {
    cellSetAboveSurface = passedCellSetPosition;
  }

  bool getCellSetPosition() const { return cellSetAboveSurface; }

  // Sets the filling fraction at given cell index.
  bool setFillingFraction(const int idx, const T fill) {
    if (idx < 0)
      return false;

    getFillingFractions()->at(idx) = fill;
    return true;
  }

  // Sets the filling fraction for cell which contains given point.
  bool setFillingFraction(const std::array<T, 3> &point, const T fill) {
    auto idx = findIndex(point);
    return setFillingFraction(idx, fill);
  }

  // Add to the filling fraction at given cell index.
  bool addFillingFraction(int idx, T fill) {
    if (idx < 0)
      return false;

    fillingFractions->at(idx) += fill;
    return true;
  }

  // Add to the filling fraction for cell which contains given point.
  bool addFillingFraction(const std::array<T, 3> &point, T fill) {
    auto idx = findIndex(point);
    return addFillingFraction(idx, fill);
  }

  // Add to the filling fraction for cell which contains given point only if the
  // cell has the specified material ID.
  bool addFillingFractionInMaterial(const std::array<T, 3> &point, T fill,
                                    int materialId) {
    auto idx = findIndex(point);
    if (getScalarData("Material")->at(idx) == materialId)
      return addFillingFraction(idx, fill);
    else
      return false;
  }

  // Write the cell set as .vtu file
  void writeVTU(std::string fileName) {
    lsVTKWriter<T>(cellGrid, fileName).apply();
  }

  // Save cell set data in simple text format
  void writeCellSetData(std::string fileName) const {
    auto numScalarData = cellGrid->getCellData().getScalarDataSize();

    std::ofstream file(fileName);
    file << numberOfCells << "\n";
    for (int i = 0; i < numScalarData; i++) {
      auto label = cellGrid->getCellData().getScalarDataLabel(i);
      file << label << ",";
    }
    file << "\n";

    for (size_t j = 0; j < numberOfCells; j++) {
      for (int i = 0; i < numScalarData; i++) {
        file << cellGrid->getCellData().getScalarData(i)->at(j) << ",";
      }
      file << "\n";
    }

    file.close();
  }

  // Read cell set data from text
  void readCellSetData(std::string fileName) {
    std::ifstream file(fileName);
    std::string line;

    if (!file.is_open()) {
      psLogger::getInstance()
          .addWarning("Could not open file " + fileName)
          .print();
      return;
    }

    std::getline(file, line);
    if (std::stoi(line) != numberOfCells) {
      psLogger::getInstance().addWarning("Incompatible cell set data.").print();
      return;
    }

    std::vector<std::string> labels;
    std::getline(file, line);
    {
      std::stringstream ss(line);
      std::string label;
      while (std::getline(ss, label, ',')) {
        labels.push_back(label);
      }
    }

    std::vector<std::vector<T> *> cellDataP;
    for (int i = 0; i < labels.size(); i++) {
      auto dataP = getScalarData(labels[i]);
      if (dataP == nullptr) {
        dataP = addScalarData(labels[i], 0.);
      }
    }

    for (int i = 0; i < labels.size(); i++) {
      cellDataP.push_back(getScalarData(labels[i]));
    }

    std::size_t j = 0;
    while (std::getline(file, line)) {
      std::stringstream ss(line);
      std::size_t i = 0;
      std::string value;
      while (std::getline(ss, value, ','))
        cellDataP[i++]->at(j) = std::stod(value);

      j++;
    }
    assert(j == numberOfCells && "Data incompatible");

    file.close();
  }

  // Clear the filling fractions
  void clear() {
    auto ff = getFillingFractions();
    std::fill(ff->begin(), ff->end(), 0.);
  }

  // Update the material IDs of the cell set. This function should be called if
  // the level sets, the cell set is made out of, have changed. This does not
  // work if the surface of the volume has changed. In this case, call the
  // funciton update surface first.
  void updateMaterials() {
    auto materialIds = getScalarData("Material");

    // create overlay material
    std::vector<psSmartPointer<lsDomain<T, D>>> levelSetsInOrder;
    auto plane = psSmartPointer<lsDomain<T, D>>::New(surface->getGrid());
    {
      T origin[D] = {0.};
      T normal[D] = {0.};
      origin[D - 1] = depth;
      normal[D - 1] = 1.;
      lsMakeGeometry<T, D>(plane,
                           psSmartPointer<lsPlane<T, D>>::New(origin, normal))
          .apply();
    }
    if (!cellSetAboveSurface)
      levelSetsInOrder.push_back(plane);
    for (auto ls : *levelSets)
      levelSetsInOrder.push_back(ls);
    if (cellSetAboveSurface)
      levelSetsInOrder.push_back(plane);

    // set up iterators for all materials
    std::vector<hrleConstDenseCellIterator<typename lsDomain<T, D>::DomainType>>
        iterators;
    for (auto it = levelSetsInOrder.begin(); it != levelSetsInOrder.end();
         ++it) {
      iterators.push_back(
          hrleConstDenseCellIterator<typename lsDomain<T, D>::DomainType>(
              (*it)->getDomain(), minIndex));
    }

    // move iterator for lowest material id and then adjust others if they are
    // needed
    unsigned cellIdx = 0;
    for (; iterators.front().getIndices() < maxIndex;
         iterators.front().next()) {
      // go over all materials
      for (unsigned materialId = 0; materialId < levelSetsInOrder.size();
           ++materialId) {

        auto &cellIt = iterators[materialId];
        cellIt.goToIndicesSequential(iterators.front().getIndices());

        // find out whether the centre of the box is inside
        T centerValue = 0.;
        for (int i = 0; i < (1 << D); ++i) {
          centerValue += cellIt.getCorner(i).getValue();
        }

        if (centerValue <= 0.) {
          bool isVoxel;
          // check if voxel is in bounds
          for (unsigned i = 0; i < (1 << D); ++i) {
            hrleVectorType<hrleIndexType, D> index;
            isVoxel = true;
            for (unsigned j = 0; j < D; ++j) {
              index[j] =
                  cellIt.getIndices(j) + cellIt.getCorner(i).getOffset()[j];
              if (index[j] > maxIndex[j]) {
                isVoxel = false;
                break;
              }
            }
          }

          if (isVoxel) {
            materialIds->at(cellIdx++) = materialId;
          }

          // jump out of material for loop
          break;
        }
      }
    }
    assert(cellIdx == numberOfCells &&
           "Cell set changed in `updateMaterials()'");
  }

  // Updates the surface of the cell set. The new surface should be below the
  // old surface as this function can only remove cells from the cell set.
  void updateSurface() {
    auto updateCellGrid = psSmartPointer<lsMesh<T>>::New();

    lsToVoxelMesh<T, D> voxelConverter(updateCellGrid);
    {
      auto plane = psSmartPointer<lsDomain<T, D>>::New(surface->getGrid());
      T origin[D] = {0.};
      T normal[D] = {0.};
      origin[D - 1] = depth;
      normal[D - 1] = 1.;

      lsMakeGeometry<T, D>(plane,
                           psSmartPointer<lsPlane<T, D>>::New(origin, normal))
          .apply();
      voxelConverter.insertNextLevelSet(plane);
    }
    voxelConverter.insertNextLevelSet(levelSets->back());
    voxelConverter.insertNextLevelSet(surface);
    voxelConverter.apply();

    auto cutMatIds = updateCellGrid->getCellData().getScalarData("Material");
    auto &hexas = cellGrid->template getElements<(1 << D)>();

    const auto nCutCells =
        updateCellGrid->template getElements<(1 << D)>().size();

    auto numScalarData = cellGrid->getCellData().getScalarDataSize();

    for (int elIdx = nCutCells - 1; elIdx >= 0; elIdx--) {
      if (cutMatIds->at(elIdx) == 2) {
        for (int i = 0; i < numScalarData; i++) {
          auto data = cellGrid->getCellData().getScalarData(i);
          data->erase(data->begin() + elIdx);
        }
        hexas.erase(hexas.begin() + elIdx);
      }
    }
    numberOfCells = hexas.size();
    surface->deepCopy(levelSets->back());

    buildBVH();
  }

  // Merge a trace path to the cell set.
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

  void buildNeighborhood() {
    const auto &cells = cellGrid->template getElements<(1 << D)>();
    const auto &nodes = cellGrid->getNodes();
    unsigned const numNodes = nodes.size();
    unsigned const numCells = cells.size();
    cellNeighbors.resize(numCells);

    std::vector<std::vector<unsigned>> nodeCellConnections(numNodes);

    // for each node, store which cells are connected with the node
    for (unsigned cellIdx = 0; cellIdx < numCells; cellIdx++) {
      for (unsigned cellNodeIdx = 0; cellNodeIdx < (1 << D); cellNodeIdx++) {
        nodeCellConnections[cells[cellIdx][cellNodeIdx]].push_back(cellIdx);
      }
    }

#pragma omp parallel for
    for (unsigned cellIdx = 0; cellIdx < numCells; cellIdx++) {
      auto coord = nodes[cells[cellIdx][0]];
      for (int i = 0; i < D; i++) {
        coord[i] += gridDelta / 2.;
        cellNeighbors[cellIdx][i] = -1;
        cellNeighbors[cellIdx][i + D] = -1;
      }

      for (unsigned cellNodeIdx = 0; cellNodeIdx < (1 << D); cellNodeIdx++) {
        auto &cellsAtNode = nodeCellConnections[cells[cellIdx][cellNodeIdx]];

        for (const auto &neighborCell : cellsAtNode) {
          if (neighborCell != cellIdx) {

            auto neighborCoord = nodes[cells[neighborCell][0]];
            for (int i = 0; i < D; i++)
              neighborCoord[i] += gridDelta / 2.;

            if (csUtil::distance(coord, neighborCoord) < gridDelta + eps) {

              for (int i = 0; i < D; i++) {
                if (coord[i] - neighborCoord[i] > gridDelta / 2.) {
                  cellNeighbors[cellIdx][i * 2] = neighborCell;
                } else if (coord[i] - neighborCoord[i] < -gridDelta / 2.) {
                  cellNeighbors[cellIdx][i * 2 + 1] = neighborCell;
                }
              }
            }
          }
        }
      }
    }
  }

  const std::array<int, 2 * D> &getNeighbors(unsigned long cellIdx) {
    assert(cellIdx < numberOfCells && "Cell idx out of bounds");
    return cellNeighbors[cellIdx];
  }

private:
  int findIndex(const csTriple<T> &point) {
    const auto &elems = cellGrid->template getElements<(1 << D)>();
    const auto &nodes = cellGrid->getNodes();
    int idx = -1;

    auto cellIds = BVH->getCellIds(point);
    if (!cellIds)
      return idx;
    for (const auto cellId : *cellIds) {
      if (isInsideVoxel(point, nodes[elems[cellId][0]])) {
        idx = cellId;
        break;
      }
    }
    return idx;
  }

  void adjustMaterialIds() {
    auto matIds = getScalarData("Material");

#pragma omp parallel for
    for (size_t i = 0; i < matIds->size(); i++) {
      if (matIds->at(i) > 0) {
        matIds->at(i) -= 1;
      }
    }
  }

  int findSurfaceHitPoint(csTriple<T> &hitPoint, const csTriple<T> &direction) {
    // find surface hitpoint
    auto idx = findIndex(hitPoint);

    if (idx > 0)
      return idx;

    auto moveDirection = multNew(direction, gridDelta / 2.);
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

  bool isInsideVoxel(const csTriple<T> &point, const csTriple<T> &cellMin) {
    if constexpr (D == 3)
      return point[0] >= cellMin[0] && point[0] <= (cellMin[0] + gridDelta) &&
             point[1] >= cellMin[1] && point[1] <= (cellMin[1] + gridDelta) &&
             point[2] >= cellMin[2] && point[2] <= (cellMin[2] + gridDelta);
    else
      return point[0] >= cellMin[0] && point[0] <= (cellMin[0] + gridDelta) &&
             point[1] >= cellMin[1] && point[1] <= (cellMin[1] + gridDelta);
  }

  void buildBVH() {
    auto &elems = cellGrid->template getElements<(1 << D)>();
    auto &nodes = cellGrid->getNodes();
    BVH->clearCellIds();

    for (size_t elemIdx = 0; elemIdx < elems.size(); elemIdx++) {
      for (size_t n = 0; n < (1 << D); n++) {
        auto &node = nodes[elems[elemIdx][n]];
        auto cell = BVH->getCellIds(node);
        if (cell == nullptr) {
          psLogger::getInstance().addError("BVH building error.").print();
        }
        cell->insert(elemIdx);
      }
    }
  }

  void calculateMinMaxIndex(
      const std::vector<psSmartPointer<lsDomain<T, D>>> &levelSetsInOrder) {
    // set to zero
    for (unsigned i = 0; i < D; ++i) {
      minIndex[i] = std::numeric_limits<hrleIndexType>::max();
      maxIndex[i] = std::numeric_limits<hrleIndexType>::lowest();
    }
    for (unsigned l = 0; l < levelSetsInOrder.size(); ++l) {
      auto &grid = levelSetsInOrder[l]->getGrid();
      auto &domain = levelSetsInOrder[l]->getDomain();
      for (unsigned i = 0; i < D; ++i) {
        minIndex[i] = std::min(minIndex[i], (grid.isNegBoundaryInfinite(i))
                                                ? domain.getMinRunBreak(i)
                                                : grid.getMinBounds(i));

        maxIndex[i] = std::max(maxIndex[i], (grid.isPosBoundaryInfinite(i))
                                                ? domain.getMaxRunBreak(i)
                                                : grid.getMaxBounds(i));
      }
    }
  }
};

#endif