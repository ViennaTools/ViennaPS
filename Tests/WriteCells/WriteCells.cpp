#include <iostream>

#include <lsBooleanOperation.hpp>
#include <lsExpand.hpp>
#include <lsMakeGeometry.hpp>
#include <lsToMesh.hpp>
#include <lsToSurfaceMesh.hpp>
#include <lsVTKWriter.hpp>

#include <csFromLevelSet.hpp>
#include <csVTKWriter.hpp>
#include <psDomain.hpp>

class myCellType : public cellBase {};

int main() {
  omp_set_num_threads(1);

  constexpr int D = 3;
  typedef double NumericType;
  typedef psDomain<myCellType, NumericType, D> psDomainType;

  psDomainType myDomain(1.0);

  // create a sphere in the level set
  NumericType origin[D] = {0., 0.};
  if (D == 3)
    origin[2] = 0;
  NumericType radius = 15.3;
  lsMakeGeometry<NumericType, D>(myDomain.getLevelSet(),
                                 lsSphere<NumericType, D>(origin, radius))
      .apply();
  origin[0] = 15.0;
  radius = 8.7;
  lsDomain<NumericType, D> secondSphere(myDomain.getLevelSet().getGrid());
  lsMakeGeometry<NumericType, D>(secondSphere,
                                 lsSphere<NumericType, D>(origin, radius))
      .apply();

  lsBooleanOperation<NumericType, D>(myDomain.getLevelSet(), secondSphere,
                                     lsBooleanOperationEnum::UNION)
      .apply();

  lsMesh mesh;
  lsToMesh<NumericType, D>(myDomain.getLevelSet(), mesh).apply();
  lsVTKWriter(mesh, "points.vtk").apply();
  lsToSurfaceMesh<NumericType, D>(myDomain.getLevelSet(), mesh).apply();
  lsVTKWriter(mesh, "surface.vtk").apply();

  csFromLevelSet<typename psDomainType::lsDomainType,
                 typename psDomainType::csDomainType>
      converter(myDomain.getLevelSet(), myDomain.getCellSet());
  // converter.setCalculateFillingFraction(false);
  converter.apply();

  csVTKWriter<myCellType, D>(myDomain.getCellSet(), "cells.vtp").writeVTP();
  csVTKWriter<myCellType, D>(myDomain.getCellSet(), "cells.vtr").apply();

  return 0;
}
