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

  // myCellType backGroundCell(1.0);

  psDomainType myDomain; //(1.0);//, backGroundCell);

  // create a sphere in the level set
  NumericType origin[D] = {15., 0.};
  if (D == 3)
    origin[2] = 0;
  NumericType radius = 8.7;
  lsMakeGeometry<NumericType, D>(
      myDomain.getLevelSets()->at(0),
      lsSmartPointer<lsSphere<NumericType, D>>::New(origin, radius))
      .apply();

  origin[0] = 0.0;
  radius = 15.3;
  auto secondSphere =
      lsSmartPointer<lsDomain<NumericType, D>>::New(myDomain.getGrid());
  lsMakeGeometry<NumericType, D>(
      secondSphere,
      lsSmartPointer<lsSphere<NumericType, D>>::New(origin, radius))
      .apply();

  lsBooleanOperation<NumericType, D>(secondSphere,
                                     myDomain.getLevelSets()->at(0),
                                     lsBooleanOperationEnum::UNION)
      .apply();

  myDomain.insertNextLevelSet(secondSphere);

  // lsExpand<NumericType, D>(myDomain.getLevelSet(), 5).apply();
  // std::cout << "width: " << myDomain.getLevelSet().getLevelSetWidth() <<
  // std::endl;
  auto mesh = lsSmartPointer<lsMesh<NumericType>>::New();
  lsToMesh<NumericType, D>(myDomain.getLevelSets()->at(0), mesh).apply();
  lsVTKWriter(mesh, "points-0.vtk").apply();
  lsToMesh<NumericType, D>(myDomain.getLevelSets()->at(1), mesh).apply();
  lsVTKWriter(mesh, "points-1.vtk").apply();
  lsToSurfaceMesh<NumericType, D>(myDomain.getLevelSets()->at(0), mesh).apply();
  lsVTKWriter(mesh, "surface-0.vtk").apply();
  lsToSurfaceMesh<NumericType, D>(myDomain.getLevelSets()->at(1), mesh).apply();
  lsVTKWriter(mesh, "surface-1.vtk").apply();

  // csFromLevelSets<typename psDomainType::lsDomainsType,
  //                 typename psDomainType::csDomainType>
  //     cellConverter(myDomain.getLevelSets(), myDomain.getCellSet());
  // // cellConverter.setCalculateFillingFraction(false);
  // cellConverter.apply();
  // myDomain.generateCellSet();
  // std::cout << myDomain.getCellSet()->getBackGroundValue() << std::endl;

  // myDomain.getCellSet()->print();
  std::cout << "Number of materials: "
            << myDomain.getCellSet()->getNumberOfMaterials() << std::endl;
  std::cout << "Converted to Cells" << std::endl;

  // myDomain.print();

  csVTKWriter<myCellType, D>(myDomain.getCellSet(), "cells.vtp").writeVTP();
  csVTKWriter<myCellType, D>(myDomain.getCellSet(), "cells.vtr").apply();

  // convert cells back into levelSets
  csToLevelSets<typename psDomainType::lsDomainsType,
                typename psDomainType::csDomainType>
      lsConverter(myDomain.getLevelSets(), myDomain.getCellSet());
  lsConverter.apply();

  lsToSurfaceMesh<NumericType, D>(myDomain.getLevelSets()->at(0), mesh).apply();
  lsVTKWriter(mesh, "newSurface-0.vtk").apply();
  lsToSurfaceMesh<NumericType, D>(myDomain.getLevelSets()->at(1), mesh).apply();
  lsVTKWriter(mesh, "newSurface-1.vtk").apply();

  lsToMesh<NumericType, D>(myDomain.getLevelSets()->at(0), mesh).apply();
  lsVTKWriter(mesh, "newPoints-0.vtk").apply();
  lsToMesh<NumericType, D>(myDomain.getLevelSets()->at(1), mesh).apply();
  lsVTKWriter(mesh, "newPoints-1.vtk").apply();

  return 0;
}
