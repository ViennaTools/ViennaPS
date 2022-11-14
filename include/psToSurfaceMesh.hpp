#pragma once

#include <lsToSurfaceMesh.hpp>
#include <psDomain.hpp>

template <class T, int D> class psToSurfaceMesh {
private:
  lsToSurfaceMesh<T, D> meshConverter;

public:
  psToSurfaceMesh(const psSmartPointer<psDomain<T, D>> passedDomain,
                  lsSmartPointer<lsMesh<T>> passedMesh, double eps = 1e-12)
      : meshConverter(passedDomain->getSurfaceLevelSet(), passedMesh, eps) {}

  void apply() { meshConverter.apply(); }

  void setDomain(const psSmartPointer<psDomain<T, D>> passedDomain) {
    meshConverter.setLevelSet(passedDomain->getSurfaceLevelSet());
  }

  void setMesh(psSmartPointer<lsMesh<T>> passedMesh) {
    meshConverter.setMesh(passedMesh);
  }
};