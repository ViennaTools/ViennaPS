#pragma once

#include <omp.h>

#include <atomic>
#include <map>
#include <utility>

#include <hrleDenseCellIterator.hpp>
#include <hrleSparseCellIterator.hpp>
#include <lsDomain.hpp>
#include <lsMarchingCubes.hpp>
#include <lsMesh.hpp>

#include <psDomain.hpp>
#include <vcKDTree.hpp>

namespace viennaps {

namespace gpu {

using namespace viennacore;

/// Extract an explicit viennals::Mesh<> instance from an viennals Domain.
/// The interface is then described by explicit surface elements:
/// Lines in 2D, Triangles in 3D.
template <class LsNumType, class MeshNumType = LsNumType, int D = 3>
class ToSurfaceMesh {
  typedef viennals::Domain<LsNumType, D> lsDomainType;
  typedef typename viennals::Domain<LsNumType, D>::DomainType hrleDomainType;
  typedef KDTree<LsNumType, std::array<LsNumType, 3>> kdTreeType;

  std::vector<SmartPointer<lsDomainType>> levelSets;
  SmartPointer<viennals::Mesh<MeshNumType>> mesh{nullptr};
  SmartPointer<kdTreeType> kdTree{nullptr};

  const MeshNumType epsilon;

public:
  ToSurfaceMesh(double eps = 1e-12) : epsilon(eps) {}

  ToSurfaceMesh(SmartPointer<lsDomainType> passedLevelSet,
                SmartPointer<viennals::Mesh<MeshNumType>> passedMesh,
                SmartPointer<kdTreeType> passedKdTree = nullptr,
                double eps = 1e-12)
      : mesh(passedMesh), epsilon(eps), kdTree(passedKdTree) {
    levelSets.push_back(passedLevelSet);
  }

  ToSurfaceMesh(SmartPointer<::viennaps::Domain<LsNumType, D>> passedDomain,
                SmartPointer<viennals::Mesh<MeshNumType>> passedMesh,
                SmartPointer<kdTreeType> passedKdTree = nullptr,
                double eps = 1e-12)
      : mesh(passedMesh), epsilon(eps), kdTree(passedKdTree) {
    for (auto &ls : passedDomain->getLevelSets()) {
      levelSets.push_back(ls);
    }
  }

  ToSurfaceMesh(SmartPointer<viennals::Mesh<MeshNumType>> passedMesh,
                SmartPointer<kdTreeType> passedKdTree = nullptr,
                double eps = 1e-12)
      : mesh(passedMesh), epsilon(eps), kdTree(passedKdTree) {}

  void insertNextLevelSet(SmartPointer<lsDomainType> passedLevelSet) {
    levelSets.push_back(passedLevelSet);
  }

  void setMesh(SmartPointer<viennals::Mesh<MeshNumType>> passedMesh) {
    mesh = passedMesh;
  }

  void setKdTree(SmartPointer<kdTreeType> passedKdTree) {
    kdTree = passedKdTree;
  }

  void apply() {
    if (levelSets.empty()) {
      Logger::getInstance()
          .addWarning("No level sets were passed to ToSurfaceMesh.")
          .print();
      return;
    }
    if (mesh == nullptr) {
      Logger::getInstance()
          .addWarning("No mesh was passed to ToSurfaceMesh.")
          .print();
      return;
    }

    mesh->clear();
    const auto gridDelta = levelSets.back()->getGrid().getGridDelta();
    const MeshNumType minNodeDistance = gridDelta / 5.;
    mesh->minimumExtent =
        std::array<MeshNumType, 3>{std::numeric_limits<MeshNumType>::max(),
                                   std::numeric_limits<MeshNumType>::max(),
                                   std::numeric_limits<MeshNumType>::max()};
    mesh->maximumExtent =
        std::array<MeshNumType, 3>{std::numeric_limits<MeshNumType>::min(),
                                   std::numeric_limits<MeshNumType>::min(),
                                   std::numeric_limits<MeshNumType>::min()};

    const unsigned int corner0[12] = {0, 1, 2, 0, 4, 5, 6, 4, 0, 1, 3, 2};
    const unsigned int corner1[12] = {1, 3, 3, 2, 5, 7, 7, 6, 4, 5, 7, 6};
    const unsigned int direction[12] = {0, 1, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2};

    // test if level set function consists of at least 2 layers of
    // defined grid points
    if (levelSets.back()->getLevelSetWidth() < 2) {
      Logger::getInstance()
          .addWarning("Levelset is less than 2 layers wide. Export might fail!")
          .print();
    }

    typedef typename std::map<hrleVectorType<hrleIndexType, D>, unsigned>
        nodeContainerType;

    nodeContainerType nodes[D];

    typename nodeContainerType::iterator nodeIt;
    lsInternal::MarchingCubes marchingCubes;

    std::vector<std::array<MeshNumType, 3>> triangleCenters;
    std::vector<std::array<MeshNumType, 3>> normals;
    const bool buildKdTreeFlag = kdTree != nullptr;

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
      for (int i = 0; i < (1 << D); i++) {
        if (cellIt.getCorner(i).getValue() >= MeshNumType(0))
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
            std::array<MeshNumType, 3> cc{}; // initialise with zeros
            for (int z = 0; z < D; z++) {
              if (z != dir) {
                // TODO might not need BitMaskToVector here, just check if z
                // bit is set
                cc[z] = static_cast<MeshNumType>(
                    cellIt.getIndices(z) +
                    BitMaskToVector<D, hrleIndexType>(p0)[z]);
              } else {
                MeshNumType d0, d1;

                d0 = static_cast<MeshNumType>(cellIt.getCorner(p0).getValue());
                d1 = static_cast<MeshNumType>(cellIt.getCorner(p1).getValue());

                // calculate the surface-grid intersection point
                if (d0 == -d1) { // includes case where d0=d1=0
                  cc[z] = static_cast<MeshNumType>(cellIt.getIndices(z)) + 0.5;
                } else {
                  if (std::abs(d0) <= std::abs(d1)) {
                    cc[z] = static_cast<MeshNumType>(cellIt.getIndices(z)) +
                            (d0 / (d0 - d1));
                  } else {
                    cc[z] = static_cast<MeshNumType>(cellIt.getIndices(z) + 1) -
                            (d1 / (d1 - d0));
                  }
                }
                cc[z] = std::max(cc[z], cellIt.getIndices(z) + epsilon);
                cc[z] = std::min(cc[z], (cellIt.getIndices(z) + 1) - epsilon);
              }
              cc[z] = gridDelta * cc[z];
            }

            int nodeIdx = checkIfNodeExists(cc, minNodeDistance);
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

        if (!triangleMisformed(nod_numbers)) {
          auto normal = calculateNormal(mesh->nodes[nod_numbers[0]],
                                        mesh->nodes[nod_numbers[1]],
                                        mesh->nodes[nod_numbers[2]]);
          LsNumType norm =
              std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] +
                        normal[2] * normal[2]);
          if (norm > gridDelta * gridDelta * 1e-4) {
            mesh->insertNextElement(nod_numbers); // insert new surface element
            for (int d = 0; d < D; d++) {
              normal[d] /= norm;
            }
            normals.push_back(normal);

            if (buildKdTreeFlag) {
              triangleCenters.push_back(
                  {static_cast<LsNumType>(mesh->nodes[nod_numbers[0]][0] +
                                          mesh->nodes[nod_numbers[1]][0] +
                                          mesh->nodes[nod_numbers[2]][0]) /
                       static_cast<LsNumType>(3.),
                   static_cast<LsNumType>(mesh->nodes[nod_numbers[0]][1] +
                                          mesh->nodes[nod_numbers[1]][1] +
                                          mesh->nodes[nod_numbers[2]][1]) /
                       static_cast<LsNumType>(3.),
                   static_cast<LsNumType>(mesh->nodes[nod_numbers[0]][2] +
                                          mesh->nodes[nod_numbers[1]][2] +
                                          mesh->nodes[nod_numbers[2]][2]) /
                       static_cast<LsNumType>(3.)});
            }
          }
        }
      }
    }

    mesh->cellData.insertNextVectorData(normals, "Normals");

    if (buildKdTreeFlag) {
      kdTree->setPoints(triangleCenters);
      kdTree->build();
    }
  }

  int checkIfNodeExists(const std::array<MeshNumType, 3> &node,
                        const MeshNumType minNodeDistance) {
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
        if (nodeClose(node, nodes[i], minNodeDistance)) {
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

  static bool nodeClose(const std::array<MeshNumType, 3> &nodeA,
                        const std::array<MeshNumType, 3> &nodeB,
                        const MeshNumType distance) {
    const auto nodeDist = std::abs(nodeA[0] - nodeB[0]) +
                          std::abs(nodeA[1] - nodeB[1]) +
                          std::abs(nodeA[2] - nodeB[2]);
    if (nodeDist < distance)
      return true;

    return false;
  }

  static bool inline triangleMisformed(
      const std::array<unsigned, D> &nod_numbers) {
    if constexpr (D == 3) {
      return nod_numbers[0] == nod_numbers[1] ||
             nod_numbers[0] == nod_numbers[2] ||
             nod_numbers[1] == nod_numbers[2];
    } else {
      return nod_numbers[0] == nod_numbers[1];
    }
  }

  std::array<MeshNumType, 3>
  calculateNormal(const std::array<MeshNumType, 3> &nodeA,
                  const std::array<MeshNumType, 3> &nodeB,
                  const std::array<MeshNumType, 3> &nodeC) {
    std::array<MeshNumType, 3> U{nodeB[0] - nodeA[0], nodeB[1] - nodeA[1],
                                 nodeB[2] - nodeA[2]};
    std::array<MeshNumType, 3> V{nodeC[0] - nodeA[0], nodeC[1] - nodeA[1],
                                 nodeC[2] - nodeA[2]};
    return {U[1] * V[2] - U[2] * V[1], U[2] * V[0] - U[0] * V[2],
            U[0] * V[1] - U[1] * V[0]};
  }
};

} // namespace gpu
} // namespace viennaps