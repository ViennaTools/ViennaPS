#pragma once

#include <lsDomain.hpp>
#include <lsToDiskMesh.hpp>

#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>

#include <psDomain.hpp>
#include <psSmartPointer.hpp>

// WARNING: THIS ONLY WORK FOR GEOMETRIES WHICH PROJECT NICELY ON THE XY PLANE.
template <class NumericType> class culsRefineMesh {
  typedef CGAL::Simple_cartesian<double> K;
  typedef CGAL::Surface_mesh<K::Point_3> SM;
  typedef boost::property_map<SM, CGAL::vertex_point_t>::type VPMap;
  typedef boost::property_map_value<SM, CGAL::vertex_point_t>::type Point_3;
  typedef boost::graph_traits<SM>::vertex_descriptor vertex_descriptor;
  typedef boost::graph_traits<SM>::edge_descriptor edge_descriptor;
  typedef boost::graph_traits<SM>::face_descriptor face_descriptor;
  typedef boost::graph_traits<SM>::halfedge_descriptor halfedge_descriptor;

  psSmartPointer<lsMesh<NumericType>> mesh = nullptr;
  double length = 1.;
  unsigned iterations = 100;

public:
  culsRefineMesh() {}
  culsRefineMesh(psSmartPointer<lsMesh<NumericType>> passedMesh,
                 const NumericType passedLength)
      : mesh(passedMesh), length(passedLength) {}

  void apply() {

    /********************************************
     * Create a SurfaceMesh from the input mesh *
     ********************************************/
    SM sm;
    VPMap vpmap = get(CGAL::vertex_point, sm);

    //  Get nb of points and cells
    std::size_t nb_points = mesh->nodes.size();
    std::size_t nb_cells = mesh->triangles.size();

    // Extract points
    std::vector<vertex_descriptor> vertex_map(nb_points);
    for (std::size_t i = 0; i < nb_points; ++i) {
      vertex_descriptor v = add_vertex(sm);
      put(vpmap, v,
          K::Point_3(mesh->nodes[i][0], mesh->nodes[i][1], mesh->nodes[i][2]));
      vertex_map[i] = v;
    }

    // Extract cells
    for (std::size_t i = 0; i < nb_cells; ++i) {
      auto &cell = mesh->triangles[i];
      std::size_t nb_vertices = cell.size();
      std::vector<vertex_descriptor> vr(nb_vertices);
      for (std::size_t k = 0; k < nb_vertices; ++k)
        vr[k] = vertex_map[cell[k]];
      CGAL::Euler::add_face(vr, sm);
    }

    std::vector<vertex_descriptor> isolated_vertices;
    for (SM::vertex_iterator vit = sm.vertices_begin();
         vit != sm.vertices_end(); ++vit) {
      if (sm.is_isolated(*vit))
        isolated_vertices.push_back(*vit);
    }

    for (std::size_t i = 0; i < isolated_vertices.size(); ++i)
      sm.remove_vertex(isolated_vertices[i]);

    if (!is_triangle_mesh(sm)) {
      psLogger::getInstance()
          .addWarning("The surface mesh must be triangulated.")
          .print();
      return;
    }

    /*****************************
     * Apply Isotropic remeshing *
     *****************************/
    CGAL::Polygon_mesh_processing::isotropic_remeshing(
        sm.faces(), length, sm,
        CGAL::Polygon_mesh_processing::parameters::number_of_iterations(
            iterations));

    /**********************************
     * Pass the SM data to the output *
     **********************************/
    std::vector<unsigned> Vids(sm.number_of_vertices());
    unsigned inum = 0;
    mesh->clear();

    for (vertex_descriptor v : vertices(sm)) {
      const K::Point_3 &p = get(vpmap, v);
      std::array<NumericType, 3> node{(NumericType)p.x(), (NumericType)p.y(),
                                      (NumericType)p.z()};
      mesh->insertNextNode(node);
      Vids[v] = inum++;
    }

    for (face_descriptor f : faces(sm)) {
      std::array<unsigned, 3> triangle;
      std::size_t n = 0;
      for (halfedge_descriptor h : halfedges_around_face(halfedge(f, sm), sm)) {
        auto vId = Vids[target(h, sm)];
        triangle[n++] = vId;
      }
      assert(n == 2 && "No triangle");

      mesh->insertNextTriangle(triangle);
    }
  }
};