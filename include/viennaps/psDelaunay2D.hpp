#pragma once

#include <lsDelaunay2D.hpp>

#include <psDomain.hpp>

namespace viennaps {

using namespace viennacore;

template <typename T> class Delaunay2D {
  using DomainType = Domain<T, 2>;

  SmartPointer<DomainType> domain_;
  SmartPointer<viennals::Mesh<T>> mesh_;
  double maxTriangleSize_ = 0.0;
  int bottomExtent_ = 0;

public:
  Delaunay2D() = default;
  //   Delaunay2D(SmartPointer<DomainType> domain, SmartPointer<Mesh<T>> mesh)
  //   : domain_(domain), mesh_(mesh) {}

  void setDomain(SmartPointer<DomainType> domain) { domain_ = domain; }

  void setMesh(SmartPointer<viennals::Mesh<T>> mesh) { mesh_ = mesh; }

  void setMaxTriangleSize(double maxTriangleSize) {
    maxTriangleSize_ = maxTriangleSize;
  }

  void setBottomExtent(int bottomExtent) { bottomExtent_ = bottomExtent; }

  void apply() {
    viennals::Delaunay2D<T> delaunay;
    delaunay.setMesh(mesh_);
    for (auto &ls : domain_->getLevelSets()) {
      delaunay.insertNextLevelSet(ls);
    }
    delaunay.setMaxTriangleSize(maxTriangleSize_);
    delaunay.setBottomExtent(bottomExtent_);
    delaunay.setMaterialMap(domain_->getMaterialMap()->getMaterialMap());
    delaunay.apply();
  }
};
} // namespace viennaps