#include <iostream>

#include <lsBooleanOperation.hpp>
#include <lsExpand.hpp>
#include <lsMakeGeometry.hpp>
#include <lsToMesh.hpp>
#include <lsToSurfaceMesh.hpp>
#include <lsVTKWriter.hpp>

#include <csFromLevelSets.hpp>
#include <csToLevelSets.hpp>
#include <csVTKWriter.hpp>
#include <psDomain.hpp>

class myCellType : public cellBase {
  using cellBase::cellBase;
};

int main() {
  omp_set_num_threads(1);

  constexpr int D = 3;
  typedef double NumericType;
  typedef psDomain<myCellType, NumericType, D> psDomainType;

  myCellType backGroundCell(1.0);

  psDomainType myDomain(1.0, backGroundCell);

  // create a sphere in the level set
  NumericType origin[D] = {15.0, 0.};
  if (D == 3)
    origin[2] = 0;
  NumericType radius = 8.7;
  lsMakeGeometry<NumericType, D>(
      myDomain.getLevelSets()[0],
      lsSmartPointer<lsSphere<NumericType, D>>::New(origin, radius))
      .apply();

  origin[0] = 0;
  radius = 15.3;
  auto secondSphere =
      lsSmartPointer<lsDomain<NumericType, D>>::New(myDomain.getGrid());
  lsMakeGeometry<NumericType, D>(
      secondSphere,
      lsSmartPointer<lsSphere<NumericType, D>>::New(origin, radius))
      .apply();


  lsBooleanOperation<NumericType, D>(secondSphere, myDomain.getLevelSets()[0],
                                     lsBooleanOperationEnum::UNION)
      .apply();
  myDomain.insertNextLevelSet(secondSphere);

  // lsExpand<NumericType, D>(myDomain.getLevelSet(), 5).apply();
  // std::cout << "width: " << myDomain.getLevelSet().getLevelSetWidth() <<
  // std::endl;
  auto mesh = lsSmartPointer<lsMesh>::New();
  lsToMesh<NumericType, D>(myDomain.getLevelSets()[0], mesh).apply();
  lsVTKWriter(mesh, "points.vtk").apply();
  lsToSurfaceMesh<NumericType, D>(myDomain.getLevelSets()[0], mesh).apply();
  lsVTKWriter(mesh, "surface0.vtk").apply();
  lsToSurfaceMesh<NumericType, D>(myDomain.getLevelSets()[1], mesh).apply();
  lsVTKWriter(mesh, "surface1.vtk").apply();

  csFromLevelSets<typename psDomainType::lsDomainsType,
                  typename psDomainType::csDomainType>
      cellConverter(myDomain.getLevelSets(), myDomain.getCellSet());
  // cellConverter.setCalculateFillingFraction(false);
  cellConverter.apply();
  // myDomain.generateCellSet();
  // std::cout << myDomain.getCellSet()->getBackGroundValue() << std::endl;

  // myDomain.getCellSet()->print();

  std::cout << "Converted to Cells" << std::endl;

  // myDomain.print();

  csVTKWriter<myCellType, D>(myDomain.getCellSet(), "cells.vtp").writeVTP();
  csVTKWriter<myCellType, D>(myDomain.getCellSet(), "cells.vtr").apply();

  // csToLevelSets<typename psDomainType::lsDomainType,
  //              typename psDomainType::csDomainType>
  //     lsConverter(myDomain.getLevelSets(), myDomain.getCellSet());
  // lsConverter.apply();

  // lsToMesh<NumericType, D>(myDomain.getLevelSet(), mesh).apply();
  // lsVTKWriter(mesh, "newPoints.vtk").apply();
  // lsToSurfaceMesh<NumericType, D>(myDomain.getLevelSet(), mesh).apply();
  // lsVTKWriter(mesh, "newSurface.vtk").apply();

  return 0;
}
