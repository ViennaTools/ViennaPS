#pragma once

#include "../psDomain.hpp"
#include "csDenseCellSet.hpp"

template <class T> class cellBase {
  std::array<T, 3> center;
  unsigned id;

public:
  cellBase(const std::array<T, 3> &passedCenter, const unsigned passedId)
      : center(passedCenter), id(passedId) {}

  std::array<T, 3> &getCenter() { return center; }
  unsigned getId() { return id; }
  virtual std::vector<T> write() const { return {}; }
};

template <class T, int D, class cellType = cellBase<T>> class csSurfaceCellSet {
  psSmartPointer<psDomain<T, D>> domain = nullptr;
  psSmartPointer<csBVH<T, D>> BVH = nullptr;

  std::vector<cellType> surfaceCells;
  std::vector<cellType> innerCells;

public:
  csSurfaceCellSet() = default;

  csSurfaceCellSet(const psSmartPointer<psDomain<T, D>> &passedDomain,
                   const T extraHeight = 0.) {
    domain = passedDomain;
    auto highestPoint = domain->getBoundingBox()[1][D - 1];
    domain->generateCellSet(highestPoint + extraHeight, psMaterial::GAS, true);
    generate();
    buildBVH();
  }

  std::vector<cellType> &getSurfaceCells() { return surfaceCells; }

  std::vector<cellType> &getInnerCells() { return innerCells; }

  cellType &getCell(const std::array<T, 3> &point) {
    auto gridDelta = domain->getCellSet()->getGridDelta();
    auto cellIds = BVH->getCellIds(point);
    if (!cellIds) {
      psLogger::getInstance()
          .addError("Point " + psUtils::arrayToString(point) +
                    " not in domain. BVH failure.")
          .print();
    } else {
      for (const auto cellId : *cellIds) {
        auto &cell = cellId < 0 ? innerCells[-cellId] : surfaceCells[cellId];
        if (isInsideVoxel(point, cell.getCenter(), gridDelta)) {
          return cell;
        }
      }
    }

    // What happened here?
    // std::vector<cellType &> getNeighbors(const cellType &cell) {
    //   std::vector<cellType &> neighbors;
    //   auto cellSet = domain->getCellSet();

    //   return neighbors;
    // }

    psLogger::getInstance()
        .addError("Point " + psUtils::arrayToString(point) + " not in domain")
        .print();
    return surfaceCells[0];
  }

  psSmartPointer<psDomain<T, D>> getDomain() { return domain; }

  void makeSurfaceCells(const std::vector<unsigned> &surfaceCellIds) {
    for (auto id : surfaceCellIds) {
      for (std::size_t i = 0; i < innerCells.size(); ++i) {
        if (innerCells[i].getId() == id) {
          surfaceCells.push_back(innerCells[i]);
          innerCells.erase(innerCells.begin() + i);
          break;
        }
      }
    }

    buildBVH();
  }

  void writeVTU(std::string fileName) {
    auto cellSet = domain->getCellSet();

    auto surfaceData = cellSet->getScalarData("SurfaceCells");
    int numData = surfaceCells[0].write().size();

    for (int i = 0; i < numData; ++i) {
      auto data = cellSet->addScalarData("values" + std::to_string(i), 0.);

      for (auto &c : surfaceCells) {
        auto writeData = c.write();
        data->at(c.getId()) = writeData[i];
      }

      for (auto &c : innerCells) {
        auto writeData = c.write();
        data->at(c.getId()) = writeData[i];
      }
    }

    cellSet->writeVTU(fileName);
  }

private:
  void generate() {
    auto cellSet = domain->getCellSet();
    cellSet->buildNeighborhood();
    auto materials = cellSet->getScalarData("Material");

    int numThreads = omp_get_max_threads();
    std::vector<std::vector<cellType>> sharedSurfaceCells(numThreads);
    std::vector<std::vector<cellType>> sharedInnerCells(numThreads);

#pragma omp parallel
    {
      int threadNum = omp_get_thread_num();
      auto localSurfaceCells = &sharedSurfaceCells[threadNum];
      auto localInnerCells = &sharedInnerCells[threadNum];

#pragma omp for
      for (int i = 0; i < materials->size(); ++i) {
        if (!psMaterialMap::isMaterial(materials->at(i), psMaterial::GAS)) {
          auto neighbors = cellSet->getNeighbors(i);
          for (auto n : neighbors) {
            if (n >= 0 &&
                psMaterialMap::isMaterial(materials->at(n), psMaterial::GAS)) {
              auto center = cellSet->getCellCenter(i);
              localSurfaceCells->emplace_back(center, i);
              break;
            }
          }
        } else {
          localInnerCells->emplace_back(cellSet->getCellCenter(i), i);
        }
      }

#pragma omp critical
      {
        surfaceCells.insert(surfaceCells.end(), localSurfaceCells->begin(),
                            localSurfaceCells->end());
        innerCells.insert(innerCells.end(), localInnerCells->begin(),
                          localInnerCells->end());
      }
    }

    auto surfaceData = cellSet->addScalarData("SurfaceCells", 0.);
    for (auto &c : surfaceCells) {
      surfaceData->at(c.getId()) = 1;
    }
  }

  void buildBVH() {
    auto cellSet = domain->getCellSet();
    auto cellGrid = cellSet->getCellGrid();
    auto gridDelta = cellSet->getGridDelta();
    T eps = 1e-6 * gridDelta;

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

    int BVHlayers = 0;
    while (minExtent / 2 > gridDelta) {
      BVHlayers++;
      minExtent /= 2;
    }

    BVH =
        psSmartPointer<csBVH<T, D>>::New(cellSet->getBoundingBox(), BVHlayers);

    psUtils::Timer timer;
    timer.start();
    auto &elems = cellGrid->template getElements<(1 << D)>();
    auto &nodes = cellGrid->getNodes();
    BVH->clearCellIds();

    for (int i = 0; i < surfaceCells.size(); ++i) {
      for (int j = 0; j < (1 << D); ++j) {
        auto &node = nodes[elems[surfaceCells[i].getId()][j]];
        BVH->getCellIds(node)->insert(i);
      }
    }

    // inner cells are stored with negative ids
    for (int i = 0; i < innerCells.size(); ++i) {
      for (int j = 0; j < (1 << D); ++j) {
        auto &node = nodes[elems[innerCells[i].getId()][j]];
        BVH->getCellIds(node)->insert(-i);
      }
    }

    timer.finish();
    psLogger::getInstance()
        .addTiming("Building surface cell set BVH took",
                   timer.currentDuration * 1e-9)
        .print();
  }

  bool isInsideVoxel(const std::array<T, 3> &point,
                     const std::array<T, 3> &center, const T gridDelta) {
    for (unsigned i = 0; i < D; ++i) {
      if (std::abs(point[i] - center[i]) > gridDelta / 2) {
        return false;
      }
    }
    return true;
  }
};
