#pragma once

#include <omp.h>

#include <atomic>
#include <iostream>
#include <map>

#include <hrleSparseCellIterator.hpp>
#include <lsDomain.hpp>
#include <lsMarchingCubes.hpp>
#include <lsMesh.hpp>
#include <utLog.hpp>

/// Extract an explicit lsMesh<> instance from an lsDomain.
/// The interface is then described by explciit surface elements:
/// Lines in 2D, Triangles in 3D.
template <class T, int D = 3> class culsToSurfaceMesh {
  typedef typename lsDomain<T, D>::DomainType hrleDomainType;

  psSmartPointer<lsDomain<double, D>> dlevelSet = nullptr;
  psSmartPointer<lsDomain<float, D>> flevelSet = nullptr;
  psSmartPointer<lsMesh<T>> mesh = nullptr;
  // std::vector<hrleIndexType> meshNodeToPointIdMapping;
  const T epsilon;
  bool updatePointData = true;
  bool useFloat;
  T minNodeDistance = 1e-4;

public:
  culsToSurfaceMesh(double eps = 1e-12) : epsilon(eps) {}

  template <class TDom>
  culsToSurfaceMesh(const psSmartPointer<lsDomain<TDom, D>> passedLevelSet,
                    psSmartPointer<lsMesh<T>> passedMesh, double eps = 1e-12)
      : mesh(passedMesh), epsilon(eps) {
    if constexpr (std::is_same_v<TDom, float>) {
      flevelSet = passedLevelSet;
      useFloat = true;
    } else if constexpr (std::is_same_v<TDom, double>) {
      dlevelSet = passedLevelSet;
      useFloat = false;
    } else {
      utLog::getInstance()
          .addWarning("Level set numeric type not compatiable.")
          .print();
    }
  }

  template <class TDom>
  void setLevelSet(psSmartPointer<lsDomain<TDom, D>> passedlsDomain) {
    if constexpr (std::is_same_v<TDom, float>) {
      flevelSet = passedlsDomain;
      useFloat = true;
    } else if constexpr (std::is_same_v<TDom, double>) {
      dlevelSet = passedlsDomain;
      useFloat = false;
    } else {
      utLog::getInstance()
          .addWarning("Level set numeric type not compatiable.")
          .print();
    }
  }

  void setMesh(psSmartPointer<lsMesh<T>> passedMesh) { mesh = passedMesh; }

  void setUpdatePointData(bool update) { updatePointData = update; }

  void apply() {
    if (useFloat)
      t_apply(flevelSet);
    else
      t_apply(dlevelSet);
  }

private:
  template <class TDom>
  void t_apply(psSmartPointer<lsDomain<TDom, D>> levelSet) {
    if (levelSet == nullptr) {
      utLog::getInstance()
          .addWarning("No level set was passed to culsToSurfaceMesh.")
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
    const auto gridDelta = levelSet->getGrid().getGridDelta();
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
    if (levelSet->getLevelSetWidth() < 2) {
      utLog::getInstance()
          .addWarning("Levelset is less than 2 layers wide. Export might fail!")
          .print();
    }

    typedef typename std::map<hrleVectorType<hrleIndexType, D>, unsigned>
        nodeContainerType;
    typedef typename lsDomain<TDom, D>::DomainType hrleDomainType;

    nodeContainerType nodes[D];

    typename nodeContainerType::iterator nodeIt;

    lsInternal::lsMarchingCubes marchingCubes;

    using DomainType = lsDomain<TDom, D>;
    using ScalarDataType = typename DomainType::PointDataType::ScalarDataType;
    using VectorDataType = typename DomainType::PointDataType::VectorDataType;

    // const bool updateData = updatePointData;
    // save how data should be transferred to new level set
    // list of indices into the old pointData vector
    // std::vector<std::vector<unsigned>> newDataSourceIds;
    // there is no multithreading here, so just use 1
    // if (updateData)
    //   newDataSourceIds.resize(1);

    // iterate over all active points
    for (hrleConstSparseCellIterator<hrleDomainType> cellIt(
             levelSet->getDomain());
         !cellIt.isFinished(); cellIt.next()) {

      for (int u = 0; u < D; u++) {
        while (!nodes[u].empty() &&
               nodes[u].begin()->first <
                   hrleVectorType<hrleIndexType, D>(cellIt.getIndices()))
          nodes[u].erase(nodes[u].begin());
      }

      unsigned signs = 0;
      for (int i = 0; i < (1 << D); i++) {
        if (cellIt.getCorner(i).getValue() >= T(0))
          signs |= (1 << i);
      }

      // all corners have the same sign, so no surface here
      if (signs == 0)
        continue;
      if (signs == (1 << (1 << D)) - 1)
        continue;

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
            std::size_t currentPointId = 0;
            for (int z = 0; z < D; z++) {
              if (z != dir) {
                // TODO might not need BitMaskToVector here, just check if z bit
                // is set
                cc[z] =
                    static_cast<T>(cellIt.getIndices(z) +
                                   BitMaskToVector<D, hrleIndexType>(p0)[z]);
              } else {
                T d0, d1;

                d0 = static_cast<T>(cellIt.getCorner(p0).getValue());
                d1 = static_cast<T>(cellIt.getCorner(p1).getValue());

                // calculate the surface-grid intersection point
                if (d0 == -d1) { // includes case where d0=d1=0
                  currentPointId = cellIt.getCorner(p0).getPointId();
                  cc[z] = static_cast<T>(cellIt.getIndices(z)) + 0.5;
                } else {
                  if (std::abs(d0) <= std::abs(d1)) {
                    currentPointId = cellIt.getCorner(p0).getPointId();
                    cc[z] =
                        static_cast<T>(cellIt.getIndices(z)) + (d0 / (d0 - d1));
                  } else {
                    currentPointId = cellIt.getCorner(p1).getPointId();
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
              if (cc[0] < mesh->minimumExtent[0])
                mesh->minimumExtent[0] = cc[0];
              if (cc[0] > mesh->maximumExtent[0])
                mesh->maximumExtent[0] = cc[0];
              if (cc[1] < mesh->minimumExtent[1])
                mesh->minimumExtent[1] = cc[1];
              if (cc[1] > mesh->maximumExtent[1])
                mesh->maximumExtent[1] = cc[1];
              if (cc[2] < mesh->minimumExtent[2])
                mesh->minimumExtent[2] = cc[2];
              if (cc[2] > mesh->maximumExtent[2])
                mesh->maximumExtent[2] = cc[2];
            }
            nodes[dir][d] = nod_numbers[n];

            // if (updateData)
            //   newDataSourceIds[0].push_back(currentPointId);
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

        if (!triangleMisformed)
          mesh->insertNextElement(nod_numbers); // insert new surface element
      }
    }

    // now copy old data into new level set
    // if (updateData) {
    //   mesh->getPointData().translateFromMultiData(levelSet->getPointData(),
    //                                               newDataSourceIds);
    // }
  }

  int checkIfNodeExists(const std::array<T, D> &node) {
    const auto &nodes = mesh->getNodes();
    const uint N = nodes.size();
    const int maxThreads = omp_get_max_threads();
    const int numThreads = maxThreads > N ? 1 : maxThreads;
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

  bool nodeClose(const std::array<T, D> &nodeA, const std::array<T, D> &nodeB) {
    const auto nodeDist = std::abs(nodeA[0] - nodeB[0]) +
                          std::abs(nodeA[1] - nodeB[1]) +
                          std::abs(nodeA[2] - nodeB[2]);
    if (nodeDist < minNodeDistance)
      return true;

    return false;
  }
};
