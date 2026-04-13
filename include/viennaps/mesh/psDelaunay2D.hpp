#pragma once

#include <lsToMultiSurfaceMesh.hpp>
#include <lsVTKWriter.hpp>
#include <lsWriteVisualizationMesh.hpp>
#include <psDomain.hpp>

#include <algorithm>
#include <limits>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>

#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Delaunay_mesh_face_base_2.h>
#include <CGAL/Delaunay_mesh_size_criteria_2.h>
#include <CGAL/Delaunay_mesher_2.h>
#include <CGAL/IO/write_VTU.h>

#include <vtkCellLocator.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridWriter.h>

#include "psConstraintCleaner.hpp"

namespace viennaps {

using namespace viennacore;

struct Mesh2DResult {
  std::vector<std::array<double, 2>> nodes;
  std::vector<std::array<unsigned, 3>> triangles;
  std::vector<std::array<unsigned, 2>> lines;
  std::vector<double> materialIds;
};

template <typename NumericType> class Delaunay2D {

  typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
  typedef CGAL::Triangulation_vertex_base_2<K> Vb;
  typedef CGAL::Delaunay_mesh_face_base_2<K> Fb;
  typedef CGAL::Triangulation_data_structure_2<Vb, Fb> Tds;
  typedef CGAL::Constrained_Delaunay_triangulation_2<K, Tds> CDT;
  typedef CGAL::Delaunay_mesh_size_criteria_2<CDT> Criteria;

  SmartPointer<viennaps::Domain<NumericType, 2>> domain;
  double maxTriangleSize = 1.;
  double minNodeDistance = 0.02;
  int bottomExtent = 1;
  int bottomLayerMaterialId = -1;
  int voidMaterialId = -1;
  bool closeDomain = true;
  bool cleanConstraints = true;
  bool verbose = false;
  NumericType constraintTargetSpacing = -1;
  NumericType constraintMergeThreshold = -1;
  NumericType constraintMinEdgeLength = -1;

private:
  auto cdtExtract(const CDT &cdt, const bool inDomain = true) {
    Mesh2DResult cdtMesh;
    std::unordered_map<const void *, unsigned> vertexToPointId;

    vertexToPointId.reserve(static_cast<std::size_t>(cdt.number_of_vertices()));
    cdtMesh.nodes.reserve(static_cast<std::size_t>(cdt.number_of_vertices()));

    // Use the address of the underlying CGAL vertex as a stable key.
    // (CGAL handles are not guaranteed to be hashable.)
    for (auto vit = cdt.finite_vertices_begin();
         vit != cdt.finite_vertices_end(); ++vit) {
      const auto &p = vit->point();
      const unsigned pid = cdtMesh.nodes.size();
      cdtMesh.nodes.push_back({CGAL::to_double(p.x()), CGAL::to_double(p.y())});
      vertexToPointId.emplace(static_cast<const void *>(&*vit), pid);
    }

    CGAL::internal::In_domain<CDT> in_domain;

    for (typename CDT::Finite_faces_iterator fit = cdt.finite_faces_begin(),
                                             end = cdt.finite_faces_end();
         fit != end; ++fit) {
      std::array<unsigned, 3> ids;
      if (!inDomain || get(in_domain, fit)) {
        bool ok = true;
        for (int i = 0; i < 3; ++i) {
          const auto vh = fit->vertex(i);
          const auto it = vertexToPointId.find(static_cast<const void *>(&*vh));
          if (it == vertexToPointId.end()) {
            ok = false;
            break;
          }
          ids[i] = it->second;
        }

        if (!ok)
          continue;

        cdtMesh.triangles.push_back(ids);
      }
    }

    for (typename CDT::Constrained_edges_iterator
             cei = cdt.constrained_edges_begin(),
             end = cdt.constrained_edges_end();
         cei != end; ++cei) {
      std::array<unsigned, 2> ids{static_cast<unsigned>(-1),
                                  static_cast<unsigned>(-1)};
      unsigned idi = 0;
      for (int i = 0; i < 3; ++i) {
        if (i != cei->second) {
          const auto vh = cei->first->vertex(i);
          const auto it = vertexToPointId.find(static_cast<const void *>(&*vh));
          if (it == vertexToPointId.end()) {
            ids[0] = ids[1] = static_cast<unsigned>(-1);
            break;
          }
          ids[idi++] = it->second;
        }
      }
      if (ids[0] != static_cast<unsigned>(-1) &&
          ids[1] != static_cast<unsigned>(-1)) {
        cdtMesh.lines.push_back(ids);
      }
    }

    return cdtMesh;
  }

public:
  Delaunay2D() = default;

  void setDomain(SmartPointer<Domain<NumericType, 2>> d) { domain = d; }

  void setTargetEdgeLengthFactor(double size) { maxTriangleSize = size; }

  void setBottomExtent(int extent) { bottomExtent = extent; }

  void setBottomMaterialId(int materialId) {
    bottomLayerMaterialId = materialId;
  }

  void setVoidMaterialId(int materialId) { voidMaterialId = materialId; }

  void setCloseDomain(bool close) { closeDomain = close; }

  /// Enable/disable constraint cleaning before CDT
  void setCleanConstraints(bool clean) { cleanConstraints = clean; }

  /// Enable verbose output
  void setVerboseOutput(bool v) { verbose = v; }

  /// Set target edge spacing for constraint cleaning (auto if < 0)
  void setConstraintTargetSpacing(NumericType spacing) {
    constraintTargetSpacing = spacing;
  }

  /// Set merge threshold for near-duplicate vertices (auto if < 0)
  void setConstraintMergeThreshold(NumericType threshold) {
    constraintMergeThreshold = threshold;
  }

  /// Set minimum edge length for constraint cleaning (auto if < 0)
  void setConstraintMinEdgeLength(NumericType length) {
    constraintMinEdgeLength = length;
  }

  void setSurfaceMeshMinNodeDistanceFactor(double distance) {
    minNodeDistance = distance;
  }

  auto apply() {
    if (!domain) {
      VIENNACORE_LOG_ERROR("No domain specified for Delaunay2D meshing!");
    }

    if (!domain->getMaterialMap()) {
      VIENNACORE_LOG_ERROR(
          "Domain material map is null! Delaunay2D meshing requires a material "
          "map to assign material IDs to the generated mesh.");
    }
    auto materialMap = domain->getMaterialMap()->getMaterialMap();

    if (Logger::hasDebug()) {
      verbose = true;
    }

    auto mesh = viennals::Mesh<NumericType>::New();
    viennals::ToMultiSurfaceMesh<NumericType, 2> mesher(minNodeDistance);
    viennals::WriteVisualizationMesh<NumericType, 2> visMesh;
    mesher.setMesh(mesh);
    if (verbose) {
      visMesh.setFileName("delaunay2D_visualization_mesh");
    } else {
      visMesh.setWriteToFile(false);
    }
    for (const auto &d : domain->getLevelSets()) {
      mesher.insertNextLevelSet(d);
      visMesh.insertNextLevelSet(d);
    }
    mesher.apply();
    visMesh.apply();
    // surface line mesh is now in mesh

    // remove normals
    mesh->getCellData().clear();

    // Clean constraints before CDT if enabled
    if (cleanConstraints) {
      ConstraintCleaner<NumericType> cleaner;
      cleaner.setPoints(mesh->nodes);
      cleaner.setEdges(mesh->lines);
      cleaner.setVerbose(verbose);

      if (constraintTargetSpacing > 0) {
        cleaner.setTargetSpacing(constraintTargetSpacing);
      }
      if (constraintMergeThreshold > 0) {
        cleaner.setMergeThreshold(constraintMergeThreshold);
      }
      if (constraintMinEdgeLength > 0) {
        cleaner.setMinEdgeLength(constraintMinEdgeLength);
      }

      cleaner.apply();
      cleaner.applyToMesh(mesh);
    }

    if (verbose) {
      viennals::VTKWriter<NumericType>(mesh, "delaunay2D_cleaned_constraints")
          .apply();
    }

    auto const minExtent = mesh->minimumExtent;
    auto const gridDelta = domain->getGridDelta();
    if (closeDomain) {
      // close mesh
      NumericType minX = std::numeric_limits<NumericType>::max();
      NumericType maxX = std::numeric_limits<NumericType>::lowest();
      NumericType minY = std::numeric_limits<NumericType>::max();

      for (const auto &p : mesh->nodes) {
        if (p[0] < minX)
          minX = p[0];
        if (p[0] > maxX)
          maxX = p[0];
        if (p[1] < minY)
          minY = p[1];
      }

      std::vector<unsigned> leftBoundaryIndices;
      std::vector<unsigned> rightBoundaryIndices;

      // use 1/10 of a grid spacing or similar as epsilon
      const double eps = 1e-6 * gridDelta;

      for (unsigned i = 0; i < mesh->nodes.size(); ++i) {
        if (std::abs(mesh->nodes[i][0] - minX) < eps) {
          leftBoundaryIndices.push_back(i);
        }
        if (std::abs(mesh->nodes[i][0] - maxX) < eps) {
          rightBoundaryIndices.push_back(i);
        }
      }

      std::sort(leftBoundaryIndices.begin(), leftBoundaryIndices.end(),
                [&](unsigned a, unsigned b) {
                  return mesh->nodes[a][1] < mesh->nodes[b][1];
                });

      std::sort(rightBoundaryIndices.begin(), rightBoundaryIndices.end(),
                [&](unsigned a, unsigned b) {
                  return mesh->nodes[a][1] > mesh->nodes[b][1];
                });

      auto p1 = mesh->insertNextNode(
          Vec3D<NumericType>{minX, minY - bottomExtent * gridDelta, 0.});
      auto p2 = mesh->insertNextNode(
          Vec3D<NumericType>{maxX, minY - bottomExtent * gridDelta, 0.});

      if (!rightBoundaryIndices.empty()) {
        for (size_t i = 0; i < rightBoundaryIndices.size() - 1; ++i) {
          mesh->insertNextLine(
              Vec2Dui{rightBoundaryIndices[i], rightBoundaryIndices[i + 1]});
        }
        mesh->insertNextLine(Vec2Dui{rightBoundaryIndices.back(), p2});
      }

      mesh->insertNextLine(Vec2Dui{p2, p1});

      if (!leftBoundaryIndices.empty()) {
        mesh->insertNextLine(Vec2Dui{p1, leftBoundaryIndices.front()});
        for (size_t i = 0; i < leftBoundaryIndices.size() - 1; ++i) {
          mesh->insertNextLine(
              Vec2Dui{leftBoundaryIndices[i], leftBoundaryIndices[i + 1]});
        }
      }
    }

    if (verbose) {
      viennals::VTKWriter<NumericType>(mesh, "delaunay2D_closed_surface_mesh")
          .apply();
    }

    // create constraints from surface mesh
    CDT cdt;
    {
      auto const numNodes = mesh->nodes.size();
      std::vector<CDT::Vertex_handle> vertexMap(numNodes);
      for (size_t i = 0; i < numNodes; ++i) {
        const auto &node = mesh->nodes[i];
        vertexMap[i] = cdt.insert(CDT::Point(node[0], node[1]));
      }

      for (const auto &line : mesh->lines) {
        cdt.insert_constraint(vertexMap[line[0]], vertexMap[line[1]]);
      }
    }

    // run meshing
    assert(
        maxTriangleSize > 0 &&
        "Target edge length factor must be positive for Delaunay2D meshing!");
    CGAL::refine_Delaunay_mesh_2(cdt, CGAL::parameters::criteria(Criteria(
                                          0.125, maxTriangleSize * gridDelta)));

    if (verbose) {
      std::fstream ofs("delaunay2D_cdt_mesh.vtu", std::ios::out);
      CGAL::IO::write_VTU(ofs, cdt);
      ofs.close();
    }

    auto rgrid = visMesh.getVolumeMesh();
    auto materials = rgrid->GetCellData()->GetArray("Material");

    auto cellLocator = vtkSmartPointer<vtkCellLocator>::New();
    cellLocator->SetDataSet(rgrid);
    cellLocator->BuildLocator();

    auto cdtMesh = cdtExtract(cdt);

    cdtMesh.materialIds.reserve(cdtMesh.triangles.size());
    for (const auto &tri : cdtMesh.triangles) {
      auto &p1 = cdtMesh.nodes[tri[0]];
      auto &p2 = cdtMesh.nodes[tri[1]];
      auto &p3 = cdtMesh.nodes[tri[2]];
      std::array<double, 3> centroid = {(p1[0] + p2[0] + p3[0]) / 3.,
                                        (p1[1] + p2[1] + p3[1]) / 3., 0.};

      /// TODO: use multi-threaded version here
      vtkIdType cellId = cellLocator->FindCell(centroid.data());

      if (cellId == -1) {
        // no cell found, determine if below level set domain
        if (p1[1] < minExtent[1] || p2[1] < minExtent[1] ||
            p3[1] < minExtent[1]) {
          // below level set domain
          if (bottomLayerMaterialId == -1) {
            cdtMesh.materialIds.push_back(materialMap->getMaterialId(0));
          } else {
            cdtMesh.materialIds.push_back(
                static_cast<NumericType>(bottomLayerMaterialId));
          }
        } else {
          // void
          cdtMesh.materialIds.push_back(voidMaterialId);
        }
      } else {
        // cell found
        NumericType materialId = materials->GetTuple1(cellId);
        materialId =
            materialMap->getMaterialId(static_cast<size_t>(materialId));
        cdtMesh.materialIds.push_back(materialId);
      }
    }

    if (verbose) {
      auto mesh = viennals::Mesh<NumericType>::New();
      for (const auto &node : cdtMesh.nodes) {
        mesh->insertNextNode(Vec3D<NumericType>{node[0], node[1], 0.});
      }
      mesh->triangles = cdtMesh.triangles;
      mesh->cellData.insertNextScalarData(cdtMesh.materialIds, "MaterialId");
      viennals::VTKWriter<NumericType>(mesh, "delaunay2D_final_mesh").apply();
    }

    return cdtMesh;
  }
};
} // namespace viennaps