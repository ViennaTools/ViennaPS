#pragma once

#include "psDomain.hpp"

#include <lsToSurfaceMesh.hpp>

template <class T, int D> class psToSurfaceMesh {
private:
  lsToSurfaceMesh<T, D> meshConverter;

public:
  psToSurfaceMesh(const psSmartPointer<psDomain<T, D>> passedDomain,
                  lsSmartPointer<lsMesh<T>> passedMesh, double eps = 1e-12)
      : meshConverter(passedDomain->getLevelSets()->back(), passedMesh, eps) {}

  void apply() { meshConverter.apply(); }

  void setDomain(const psSmartPointer<psDomain<T, D>> passedDomain) {
    meshConverter.setLevelSet(passedDomain->getLevelSets()->back());
  }

  void setMesh(psSmartPointer<lsMesh<T>> passedMesh) {
    meshConverter.setMesh(passedMesh);
  }
};
