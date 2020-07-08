#include <iostream>

#include <lsBooleanOperation.hpp>
#include <lsExpand.hpp>
#include <lsMakeGeometry.hpp>
#include <lsToMesh.hpp>
#include <lsToSurfaceMesh.hpp>
#include <lsVTKWriter.hpp>

#include <csFromLevelSets.hpp>
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
  lsMakeGeometry<NumericType, D>(myDomain.getLevelSets()[0],
                                 lsSmartPointer<lsSphere<NumericType, D>>::New(origin, radius))
      .apply();
  origin[0] = 15.0;
  radius = 8.7;
  auto secondSphere = lsSmartPointer<lsDomain<NumericType, D>>::New(myDomain.getGrid());
  lsMakeGeometry<NumericType, D>(secondSphere,
                                 lsSmartPointer<lsSphere<NumericType, D>>::New(origin, radius))
      .apply();

  lsBooleanOperation<NumericType, D>(myDomain.getLevelSets()[0], secondSphere,
                                     lsBooleanOperationEnum::UNION)
      .apply();

  auto mesh = lsSmartPointer<lsMesh>::New();
  lsToMesh<NumericType, D>(myDomain.getLevelSets()[0], mesh).apply();
  lsVTKWriter(mesh, "points.vtk").apply();
  lsToSurfaceMesh<NumericType, D>(myDomain.getLevelSets()[0], mesh).apply();
  lsVTKWriter(mesh, "surface.vtk").apply();

  csFromLevelSets<typename psDomainType::lsDomainsType,
                 typename psDomainType::csDomainType>
      converter(myDomain.getLevelSets(), myDomain.getCellSet());
  // converter.setCalculateFillingFraction(false);
  converter.apply();

  csVTKWriter<myCellType, D>(myDomain.getCellSet(), "cells.vtp").writeVTP();
  csVTKWriter<myCellType, D>(myDomain.getCellSet(), "cells.vtr").apply();

  return 0;
}
