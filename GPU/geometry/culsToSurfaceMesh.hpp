#pragma once

#include <omp.h>

#include <atomic>
#include <iostream>
#include <map>
#include <utility>

#include <hrleDenseCellIterator.hpp>
#include <hrleSparseCellIterator.hpp>
#include <lsDomain.hpp>
#include <lsMarchingCubes.hpp>
#include <lsMesh.hpp>
#include <utLog.hpp>

#include <psDomain.hpp>
#include <psKDTree.hpp>

/// Extract an explicit lsMesh<> instance from an lsDomain.
/// The interface is then described by explciit surface elements:
/// Lines in 2D, Triangles in 3D.
template <class T, int D = 3> class culsToSurfaceMesh {
  typedef typename lsDomain<T, D>::DomainType hrleDomainType;

  std::vector<psSmartPointer<lsDomain<double, D>>> dlevelSet;
  std::vector<psSmartPointer<lsDomain<float, D>>> flevelSet;
  psSmartPointer<lsMesh<T>> mesh{nullptr};
  psSmartPointer<psKDTree<T, std::array<T, 3>>> kdTree{nullptr};
  psSmartPointer<psMaterialMap> matMap{nullptr};

  const T epsilon;
  bool updatePointData = true;
  bool buildKdTree = false;
  bool useFloat;
  T minNodeDistance = 1e-4;
  static constexpr double wrappingLayerEpsilon = 1e-4;

public:
  culsToSurfaceMesh(double eps = 1e-12) : epsilon(eps) {}

  template <class TDom>
  culsToSurfaceMesh(psSmartPointer<lsDomain<TDom, D>> passedLevelSet,
                    psSmartPointer<lsMesh<T>> passedMesh, double eps = 1e-12)
      : mesh(passedMesh), epsilon(eps) {
    if constexpr (std::is_same_v<TDom, float>) {
      flevelSet.push_back(passedLevelSet);
      useFloat = true;
    } else if constexpr (std::is_same_v<TDom, double>) {
      dlevelSet.push_back(passedLevelSet);
      useFloat = false;
    }
  }

  template <class TDom>
  culsToSurfaceMesh(psSmartPointer<lsDomain<TDom, D>> passedLevelSet,
                    psSmartPointer<lsMesh<T>> passedMesh,
                    psSmartPointer<psKDTree<T, std::array<T, 3>>> passedKdTree,
                    double eps = 1e-12)
      : mesh(passedMesh), epsilon(eps), buildKdTree(true),
        kdTree(passedKdTree) {
    if constexpr (std::is_same_v<TDom, float>) {
      flevelSet.push_back(passedLevelSet);
      useFloat = true;
    } else if constexpr (std::is_same_v<TDom, double>) {
      dlevelSet.push_back(passedLevelSet);
      useFloat = false;
    }
  }

  template <class TDom>
  culsToSurfaceMesh(psSmartPointer<psDomain<TDom, D>> passedDomain,
                    psSmartPointer<lsMesh<T>> passedMesh,
                    psSmartPointer<psKDTree<T, std::array<T, 3>>> passedKdTree,
                    double eps = 1e-12)
      : mesh(passedMesh), epsilon(eps), buildKdTree(true), kdTree(passedKdTree),
        matMap(passedDomain->getMaterialMap()) {
    if constexpr (std::is_same_v<TDom, float>) {
      for (auto &ls : *passedDomain->getLevelSets()) {
        flevelSet.push_back(ls);
      }
      useFloat = true;
    } else if constexpr (std::is_same_v<TDom, double>) {
      for (auto &ls : *passedDomain->getLevelSets()) {
        dlevelSet.push_back(ls);
      }
      useFloat = false;
    }
  }

  template <class TDom>
  culsToSurfaceMesh(psSmartPointer<psDomain<TDom, D>> passedDomain,
                    psSmartPointer<lsMesh<T>> passedMesh, double eps = 1e-12)
      : mesh(passedMesh), epsilon(eps), buildKdTree(false),
        matMap(passedDomain->getMaterialMap()) {
    if constexpr (std::is_same_v<TDom, float>) {
      for (auto &ls : *passedDomain->getLevelSets()) {
        flevelSet.push_back(ls);
      }
      useFloat = true;
    } else if constexpr (std::is_same_v<TDom, double>) {
      for (auto &ls : *passedDomain->getLevelSets()) {
        dlevelSet.push_back(ls);
      }
      useFloat = false;
    }
  }

  culsToSurfaceMesh(psSmartPointer<lsMesh<T>> passedMesh, double eps = 1e-12)
      : mesh(passedMesh), epsilon(eps) {}

  culsToSurfaceMesh(psSmartPointer<lsMesh<T>> passedMesh,
                    psSmartPointer<psKDTree<T, std::array<T, 3>>> passedKdTree,
                    double eps = 1e-12)
      : mesh(passedMesh), epsilon(eps), buildKdTree(true),
        kdTree(passedKdTree) {}

  template <class TDom>
  void insertNextLevelSet(psSmartPointer<lsDomain<TDom, D>> passedLevelSet) {
    if constexpr (std::is_same_v<TDom, float>) {
      flevelSet.push_back(passedLevelSet);
      useFloat = true;
    } else if constexpr (std::is_same_v<TDom, double>) {
      dlevelSet.push_back(passedLevelSet);
      useFloat = false;
    }
  }

  void setMesh(psSmartPointer<lsMesh<T>> passedMesh) { mesh = passedMesh; }

  void setKdTree(psSmartPointer<psKDTree<T, std::array<T, 3>>> passedKdTree) {
    kdTree = passedKdTree;
    buildKdTree = true;
  }

  void apply() {
    if (useFloat)
      t_apply(flevelSet);
    else
      t_apply(dlevelSet);
  }

private:
  template <class TDom>
  void t_apply(std::vector<psSmartPointer<lsDomain<TDom, D>>> &levelSets) {
    if (levelSets.empty()) {
      utLog::getInstance()
          .addWarning("No level sets were passed to culsToSurfaceMesh.")
          .print();
      return;
    }
    if (mesh == nullptr) {
      utLog::getInstance()
          .addWarning("No mesh was passed to culsToSurfaceMesh.")
          .print();
      return;
    }

    mesh->clear();
    const auto gridDelta = levelSets.back()->getGrid().getGridDelta();
    minNodeDistance = gridDelta / 5.;
    mesh->minimumExtent = std::array<T, 3>{std::numeric_limits<T>::max(),
                                           std::numeric_limits<T>::max(),
                                           std::numeric_limits<T>::max()};
    mesh->maximumExtent = std::array<T, 3>{std::numeric_limits<T>::min(),
                                           std::numeric_limits<T>::min(),
                                           std::numeric_limits<T>::min()};

    const unsigned int corner0[12] = {0, 1, 2, 0, 4, 5, 6, 4, 0, 1, 3, 2};
    const unsigned int corner1[12] = {1, 3, 3, 2, 5, 7, 7, 6, 4, 5, 7, 6};
    const unsigned int direction[12] = {0, 1, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2};

    // test if level set function consists of at least 2 layers of
    // defined grid points
    if (levelSets.back()->getLevelSetWidth() < 2) {
      utLog::getInstance()
          .addWarning("Levelset is less than 2 layers wide. Export might fail!")
          .print();
    }

    typedef typename std::map<hrleVectorType<hrleIndexType, D>, unsigned>
        nodeContainerType;

    nodeContainerType nodes[D];

    typename nodeContainerType::iterator nodeIt;
    typedef typename lsDomain<TDom, D>::DomainType hrleDomainType;

    lsInternal::lsMarchingCubes marchingCubes;

    using DomainType = lsDomain<TDom, D>;
    using ScalarDataType = typename DomainType::PointDataType::ScalarDataType;
    using VectorDataType = typename DomainType::PointDataType::VectorDataType;

    hrleVectorType<hrleIndexType, D> minIndex;
    // set to zero
    for (unsigned i = 0; i < D; ++i) {
      minIndex[i] = std::numeric_limits<hrleIndexType>::max();
    }
    for (unsigned l = 0; l < levelSets.size(); ++l) {
      auto &grid = levelSets[l]->getGrid();
      auto &domain = levelSets[l]->getDomain();
      for (unsigned i = 0; i < D; ++i) {
        minIndex[i] = std::min(minIndex[i], (grid.isNegBoundaryInfinite(i))
                                                ? domain.getMinRunBreak(i)
                                                : grid.getMinBounds(i));
      }
    }

    // set up iterators for all materials
    std::vector<hrleConstDenseCellIterator<hrleDomainType>> denseIterators;
    for (const auto levelSet : levelSets) {
      denseIterators.push_back(hrleConstDenseCellIterator<hrleDomainType>(
          levelSet->getDomain(), minIndex));
    }

    std::vector<std::array<T, 3>> triangleCenters;
    const bool buildKdTreeFlag = buildKdTree;

    std::vector<T> materialIds;

    // iterate over all active surface points
    for (hrleConstSparseCellIterator<hrleDomainType> cellIt(
             levelSets.back()->getDomain());
         !cellIt.isFinished(); cellIt.next()) {

      for (int u = 0; u < D; u++) {
        while (!nodes[u].empty() &&
               nodes[u].begin()->first <
                   hrleVectorType<hrleIndexType, D>(cellIt.getIndices()))
          nodes[u].erase(nodes[u].begin());
      }

      unsigned signs = 0;
      T value = 0;
      unsigned undefined = 0;
      for (int i = 0; i < (1 << D); i++) {
        auto cVal = cellIt.getCorner(i).getValue();

        if (!cellIt.getCorner(i).isDefined()) {
          undefined++;
        } else {
          value += cVal;
        }

        if (cVal >= T(0))
          signs |= (1 << i);
      }

      // all corners have the same sign, so no surface here
      if (signs == 0)
        continue;
      if (signs == (1 << (1 << D)) - 1)
        continue;

      // go over all materials
      unsigned currentMatId = levelSets.size() - 1;
      for (unsigned materialId = 0; materialId < levelSets.size();
           ++materialId) {
        auto &matCellIt = denseIterators[materialId];
        matCellIt.goToIndicesSequential(cellIt.getIndices());
        if (!matCellIt.isDefined()) {
          continue;
        }

        T valueSum = 0;
        unsigned undefinedValues = 0;
        for (int i = 0; i < (1 << D); i++) {
          if (!matCellIt.getCorner(i).isDefined()) {
            undefinedValues++;
            continue;
          }
          valueSum += matCellIt.getCorner(i).getValue();
        }

        // std::cout << undefinedValues << " " << undefined << std::endl;
        // std::cout << valueSum << " " << value + wrappingLayerEpsilon
        // << std::endl;
        if (undefinedValues <= undefined &&
            valueSum <= value + wrappingLayerEpsilon) {
          currentMatId = materialId;
          std::cout << "Mask" << std::endl;
          break;
        }
      }

      // for each element
      const int *Triangles = (D == 2) ? marchingCubes.polygonize2d(signs)
                                      : marchingCubes.polygonize3d(signs);

      for (; Triangles[0] != -1; Triangles += D) {
        std::array<unsigned, D> nod_numbers;

        // for each node
        for (int n = 0; n < D; n++) {
          const int edge = Triangles[n];

          unsigned p0 = corner0[edge];
          unsigned p1 = corner1[edge];

          // determine direction of edge
          int dir = direction[edge];

          // look for existing surface node
          hrleVectorType<hrleIndexType, D> d(cellIt.getIndices());
          d += BitMaskToVector<D, hrleIndexType>(p0);

          nodeIt = nodes[dir].find(d);
          if (nodeIt != nodes[dir].end()) {
            nod_numbers[n] = nodeIt->second;
          } else { // if node does not exist yet
            // calculate coordinate of new node
            std::array<T, 3> cc{}; // initialise with zeros
            for (int z = 0; z < D; z++) {
              if (z != dir) {
                // TODO might not need BitMaskToVector here, just check if z
                // bit is set
                cc[z] =
                    static_cast<T>(cellIt.getIndices(z) +
                                   BitMaskToVector<D, hrleIndexType>(p0)[z]);
              } else {
                T d0, d1;

                d0 = static_cast<T>(cellIt.getCorner(p0).getValue());
                d1 = static_cast<T>(cellIt.getCorner(p1).getValue());

                // calculate the surface-grid intersection point
                if (d0 == -d1) { // includes case where d0=d1=0
                  cc[z] = static_cast<T>(cellIt.getIndices(z)) + 0.5;
                } else {
                  if (std::abs(d0) <= std::abs(d1)) {
                    cc[z] =
                        static_cast<T>(cellIt.getIndices(z)) + (d0 / (d0 - d1));
                  } else {
                    cc[z] = static_cast<T>(cellIt.getIndices(z) + 1) -
                            (d1 / (d1 - d0));
                  }
                }
                cc[z] = std::max(cc[z], cellIt.getIndices(z) + epsilon);
                cc[z] = std::min(cc[z], (cellIt.getIndices(z) + 1) - epsilon);
              }
              cc[z] = gridDelta * cc[z];
            }

            int nodeIdx = checkIfNodeExists(cc);
            if (nodeIdx >= 0) {
              // node exists or close node exists
              nod_numbers[n] = nodeIdx;
            } else {
              // insert new node
              nod_numbers[n] =
                  mesh->insertNextNode(cc); // insert new surface node
              for (int a = 0; a < D; a++) {
                if (cc[a] < mesh->minimumExtent[a])
                  mesh->minimumExtent[a] = cc[a];
                if (cc[a] > mesh->maximumExtent[a])
                  mesh->maximumExtent[a] = cc[a];
              }
            }
          }
        }

        bool triangleMisformed = false;
        if constexpr (D == 3) {
          triangleMisformed = nod_numbers[0] == nod_numbers[1] ||
                              nod_numbers[0] == nod_numbers[2] ||
                              nod_numbers[1] == nod_numbers[2];
        } else {
          triangleMisformed = nod_numbers[0] == nod_numbers[1];
        }

        if (!triangleMisformed) {
          mesh->insertNextElement(nod_numbers); // insert new surface element
          if (matMap) {
            materialIds.push_back(
                matMap->getMaterialMap()->getMaterialId(currentMatId));
          } else {
            materialIds.push_back(currentMatId);
          }
          if (buildKdTreeFlag) {
            triangleCenters.push_back({(mesh->nodes[nod_numbers[0]][0] +
                                        mesh->nodes[nod_numbers[1]][0] +
                                        mesh->nodes[nod_numbers[2]][0]) /
                                           3.f,
                                       (mesh->nodes[nod_numbers[0]][1] +
                                        mesh->nodes[nod_numbers[1]][1] +
                                        mesh->nodes[nod_numbers[2]][1]) /
                                           3.f,
                                       (mesh->nodes[nod_numbers[0]][2] +
                                        mesh->nodes[nod_numbers[1]][2] +
                                        mesh->nodes[nod_numbers[2]][2]) /
                                           3.f});
          }
        }
      }
    }

    if (buildKdTreeFlag) {
      std::cout << triangleCenters.size() << std::endl;
      kdTree->setPoints(triangleCenters);
      kdTree->build();
    }

    mesh->getCellData().insertNextScalarData(materialIds, "Material");
  }

  int checkIfNodeExists(const std::array<T, 3> &node) {
    const auto &nodes = mesh->getNodes();
    const uint N = nodes.size();
    const int maxThreads = omp_get_max_threads();
    const int numThreads = maxThreads > N / (10 * maxThreads) ? 1 : maxThreads;
    std::vector<int> threadLocal(numThreads, -1);
    std::atomic<bool> go(true);
    const uint share = N / numThreads;

#pragma omp parallel shared(node, nodes, threadLocal) num_threads(numThreads)
    {
      int threadId = omp_get_thread_num();

      uint i = threadId * share;
      const uint stop = threadId == numThreads - 1 ? N : (threadId + 1) * share;

      while (i < stop && go) {
        if (nodeClose(node, nodes[i])) {
          threadLocal[threadId] = i;
          go = false;
        }
        i++;
      }
    }

    int idx = -1;
    for (size_t i = 0; i < threadLocal.size(); i++) {
      if (threadLocal[i] >= 0) {
        idx = threadLocal[i];
        break;
      }
    }

    return idx;
  }

  bool nodeClose(const std::array<T, 3> &nodeA, const std::array<T, 3> &nodeB) {
    const auto nodeDist = std::abs(nodeA[0] - nodeB[0]) +
                          std::abs(nodeA[1] - nodeB[1]) +
                          std::abs(nodeA[2] - nodeB[2]);
    if (nodeDist < minNodeDistance)
      return true;

    return false;
  }
};
