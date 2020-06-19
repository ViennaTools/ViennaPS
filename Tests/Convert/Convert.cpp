#include <iostream>

#include <lsBooleanOperation.hpp>
#include <lsExpand.hpp>
#include <lsMakeGeometry.hpp>
#include <lsToMesh.hpp>
#include <lsToSurfaceMesh.hpp>
#include <lsVTKWriter.hpp>

#include <csFromLevelSet.hpp>
#include <csToLevelSet.hpp>
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

  // lsExpand<NumericType, D>(myDomain.getLevelSet(), 5).apply();
  // std::cout << "width: " << myDomain.getLevelSet().getLevelSetWidth() <<
  // std::endl;
  lsMesh mesh;
  lsToMesh<NumericType, D>(myDomain.getLevelSet(), mesh).apply();
  lsVTKWriter(mesh, "points.vtk").apply();
  lsToSurfaceMesh<NumericType, D>(myDomain.getLevelSet(), mesh).apply();
  lsVTKWriter(mesh, "surface.vtk").apply();

  csFromLevelSet<typename psDomainType::lsDomainType,
                 typename psDomainType::csDomainType>
      cellConverter(myDomain.getLevelSet(), myDomain.getCellSet());
  // cellConverter.setCalculateFillingFraction(false);
  cellConverter.apply();
  // myDomain.generateCellSet();

  std::cout << "Converted to Cells" << std::endl;

  // myDomain.print();

  csVTKWriter<myCellType, D>(myDomain.getCellSet(), "cells.vtp").writeVTP();
  csVTKWriter<myCellType, D>(myDomain.getCellSet(), "cells.vtr").apply();

  csToLevelSet<typename psDomainType::lsDomainType,
               typename psDomainType::csDomainType>
      lsConverter(myDomain.getLevelSet(), myDomain.getCellSet());
  lsConverter.apply();

  lsToMesh<NumericType, D>(myDomain.getLevelSet(), mesh).apply();
  lsVTKWriter(mesh, "newPoints.vtk").apply();
  lsToSurfaceMesh<NumericType, D>(myDomain.getLevelSet(), mesh).apply();
  lsVTKWriter(mesh, "newSurface.vtk").apply();

  return 0;
}
