#pragma once

#include "psProcessContext.hpp"

#include <lsMesh.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <vector>

namespace viennaps {

using namespace viennacore;

// The dual contouring mesh of a Euclidean surface, which the triangle flux
// engines trace, with the material behind each element; elements without
// area, which dual contouring makes where three vertices lie on one sharp
// edge, are left out, since the flux of an element is its number of hits per
// area. In 3D the element centres go into the kd-tree, in 2D the engines
// build it from the triangles the lines become.
// Dual contouring reaches half a grid spacing beyond a reflective boundary
// with mirrored vertex copies, with a ragged edge. The elements are clipped
// at the boundary planes of the grid, which are the extents of the mesh and
// so the reflecting walls of the ray tracers: with the walls at the
// outermost vertex, rays would leave the domain through the slit between
// the edge of the mesh and the wall.
template <class NumericType, int D, class KDTree>
void dualContouringSurface(ProcessContext<NumericType, D> &context,
                           SmartPointer<viennals::Mesh<float>> &surfaceMesh,
                           SmartPointer<KDTree> &elementKdTree,
                           std::vector<int> &materialIds) {
  const auto mesh = context.domain->getSurfaceMesh(false);
  const auto &nodes = mesh->getNodes();
  const auto *normals = mesh->getNormals();
  const auto *materials = mesh->getMaterialIds();
  const NumericType gridDelta = context.domain->getGridDelta();
  assert(normals != nullptr && materials != nullptr);

  // the boundary planes of the reflective directions
  const auto &grid = context.domain->getGrid();
  std::array<bool, 3> bounded{false, false, false};
  std::array<NumericType, 3> low{0., 0., 0.}, high{0., 0., 0.};
  for (int i = 0; i < D; ++i)
    if (grid.isBoundaryReflective(i)) {
      bounded[i] = true;
      low[i] = grid.getMinGridPoint(i) * gridDelta;
      high[i] = grid.getMaxGridPoint(i) * gridDelta;
    }
  const NumericType tolerance = NumericType(1e-9) * gridDelta;
  auto outside = [&](const Vec3D<NumericType> &x) {
    for (int i = 0; i < D; ++i)
      if (bounded[i] &&
          (x[i] < low[i] - tolerance || x[i] > high[i] + tolerance))
        return true;
    return false;
  };

  surfaceMesh->clear();
  surfaceMesh->minimumExtent = Vec3D<float>{std::numeric_limits<float>::max(),
                                            std::numeric_limits<float>::max(),
                                            std::numeric_limits<float>::max()};
  surfaceMesh->maximumExtent =
      Vec3D<float>{std::numeric_limits<float>::lowest(),
                   std::numeric_limits<float>::lowest(),
                   std::numeric_limits<float>::lowest()};
  materialIds.clear();
  std::vector<Vec3D<float>> elementNormals;
  std::vector<Vec3D<NumericType>> centers;
  std::vector<long> nodeIds(nodes.size(), -1);
  auto addPoint = [&](const Vec3D<NumericType> &x) {
    const Vec3D<float> point{static_cast<float>(x[0]), static_cast<float>(x[1]),
                             static_cast<float>(x[2])};
    const unsigned id = surfaceMesh->insertNextNode(point);
    for (int a = 0; a < D; ++a) {
      surfaceMesh->minimumExtent[a] =
          std::min(surfaceMesh->minimumExtent[a], point[a]);
      surfaceMesh->maximumExtent[a] =
          std::max(surfaceMesh->maximumExtent[a], point[a]);
    }
    return id;
  };
  // the new corners of clipped elements, by position
  std::map<std::array<long long, 3>, unsigned> clippedIds;
  auto clippedNode = [&](const Vec3D<NumericType> &x) {
    std::array<long long, 3> key;
    for (int a = 0; a < 3; ++a)
      key[a] = std::llround(x[a] / (NumericType(1e-6) * gridDelta));
    auto it = clippedIds.find(key);
    if (it != clippedIds.end())
      return it->second;
    const unsigned id = addPoint(x);
    clippedIds.emplace(key, id);
    return id;
  };
  auto degenerate = [&](const std::array<Vec3D<NumericType>, D> &x) {
    if constexpr (D == 2) {
      return Norm(x[1] - x[0]) <= NumericType(1e-10) * gridDelta;
    } else {
      NumericType longest = 0.;
      for (int k = 0; k < 3; ++k)
        longest = std::max(longest, Norm(x[(k + 1) % 3] - x[k]));
      const NumericType twiceArea =
          Norm(CrossProduct(x[1] - x[0], x[2] - x[0]));
      return twiceArea <= NumericType(1e-10) * longest * longest;
    }
  };
  auto addElement = [&](const std::array<unsigned, D> &ids,
                        const std::array<Vec3D<NumericType>, D> &x,
                        std::size_t e) {
    surfaceMesh->insertNextElement(ids);
    Vec3D<NumericType> center{0., 0., 0.};
    for (int k = 0; k < D; ++k)
      center = center + x[k] / static_cast<NumericType>(D);
    const auto &normal = (*normals)[e];
    elementNormals.push_back(Vec3D<float>{static_cast<float>(normal[0]),
                                          static_cast<float>(normal[1]),
                                          static_cast<float>(normal[2])});
    materialIds.push_back(static_cast<int>((*materials)[e]));
    centers.push_back(center);
  };

  const auto &elements = mesh->template getElements<D>();
  for (std::size_t e = 0; e < elements.size(); ++e) {
    const auto &element = elements[e];
    std::array<Vec3D<NumericType>, D> x;
    bool anyOutside = false;
    for (int k = 0; k < D; ++k) {
      x[k] = nodes[element[k]];
      anyOutside = anyOutside || outside(x[k]);
    }
    if (degenerate(x))
      continue;
    if (!anyOutside) {
      std::array<unsigned, D> ids{};
      for (int k = 0; k < D; ++k) {
        const auto node = element[k];
        if (nodeIds[node] < 0)
          nodeIds[node] = addPoint(nodes[node]);
        ids[k] = static_cast<unsigned>(nodeIds[node]);
      }
      addElement(ids, x, e);
      continue;
    }
    // clipped at every boundary plane (Sutherland-Hodgman); a line is a
    // polygon of two corners
    std::vector<Vec3D<NumericType>> poly(x.begin(), x.end());
    for (int i = 0; i < D && !poly.empty(); ++i) {
      if (!bounded[i])
        continue;
      for (int side = 0; side < 2 && !poly.empty(); ++side) {
        const NumericType plane = side == 0 ? low[i] : high[i];
        auto in = [&](const Vec3D<NumericType> &p) {
          return side == 0 ? p[i] >= plane - tolerance
                           : p[i] <= plane + tolerance;
        };
        std::vector<Vec3D<NumericType>> out;
        if constexpr (D == 2) {
          const auto &a = poly[0], &b = poly[1];
          const bool ia = in(a), ib = in(b);
          if (ia && ib) {
            out = poly;
          } else if (ia != ib) {
            const NumericType t = (plane - a[i]) / (b[i] - a[i]);
            Vec3D<NumericType> c = a + (b - a) * t;
            c[i] = plane;
            out = ia ? std::vector<Vec3D<NumericType>>{a, c}
                     : std::vector<Vec3D<NumericType>>{c, b};
          }
        } else {
          const std::size_t n = poly.size();
          for (std::size_t k = 0; k < n; ++k) {
            const auto &a = poly[k], &b = poly[(k + 1) % n];
            const bool ia = in(a), ib = in(b);
            if (ia)
              out.push_back(a);
            if (ia != ib) {
              const NumericType t = (plane - a[i]) / (b[i] - a[i]);
              Vec3D<NumericType> c = a + (b - a) * t;
              c[i] = plane;
              out.push_back(c);
            }
          }
        }
        poly.swap(out);
      }
    }
    if (poly.size() < std::size_t(D))
      continue;
    for (std::size_t k = 1; k + (D - 2) < poly.size(); ++k) {
      std::array<Vec3D<NumericType>, D> piece;
      if constexpr (D == 2) {
        piece = {poly[0], poly[1]};
      } else {
        piece = {poly[0], poly[k], poly[k + 1]};
      }
      if (degenerate(piece))
        continue;
      std::array<unsigned, D> ids{};
      for (int c = 0; c < D; ++c)
        ids[c] = clippedNode(piece[c]);
      addElement(ids, piece, e);
      if constexpr (D == 2)
        break;
    }
  }
  for (int i = 0; i < D; ++i)
    if (bounded[i]) {
      surfaceMesh->minimumExtent[i] = static_cast<float>(low[i]);
      surfaceMesh->maximumExtent[i] = static_cast<float>(high[i]);
    }
  surfaceMesh->cellData.insertNextVectorData(elementNormals, "Normals");
  if constexpr (D == 3) {
    elementKdTree->setPoints(centers);
    elementKdTree->build();
  }
}

} // namespace viennaps
