#pragma once

#include <lsDelaunay2D.hpp>

#include <psDomain.hpp>

namespace viennaps {

using namespace viennacore;

template <typename T> class Delaunay2D : public viennals::Delaunay2D<T> {
public:
  Delaunay2D() = default;
  Delaunay2D(SmartPointer<viennals::Mesh<T>> mesh)
      : viennals::Delaunay2D<T>(mesh) {}

  void setDomain(SmartPointer<Domain<T, 2>> domain) {
    this->clear();
    for (const auto &d : domain->getLevelSets()) {
      this->insertNextLevelSet(d);
    }
    this->setMaterialMap(domain->getMaterialMap()->getMaterialMap());
  }

  void setBottomMaterial(Material bottomMaterial) {
    this->setBottomLayerMaterialId(static_cast<int>(bottomMaterial));
  }

  void setVoidMaterial(Material voidMaterial) {
    this->setVoidMaterialId(static_cast<int>(voidMaterial));
  }
};
} // namespace viennaps