#pragma once

#include <lsDomain.hpp>
#include <lsToDiskMesh.hpp>

#include <psDomain.hpp>

#include <vtkCellArray.h>
#include <vtkCellLocator.h>
#include <vtkDelaunay2D.h>
#include <vtkLine.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkTriangle.h>

namespace viennaps {

using namespace viennacore;

// WARNING: THIS ONLY WORKS FOR GEOMETRIES WHICH PROJECT NICELY ON THE XY PLANE.
template <class NumericType> class DelaunaySurfaceMesh {
  using lsDomainType = SmartPointer<viennals::Domain<NumericType, 3>>;
  using psDomainType = SmartPointer<::viennaps::Domain<NumericType, 3>>;
  using MeshType = SmartPointer<viennals::Mesh<NumericType>>;

  lsDomainType levelSet = nullptr;
  MeshType surfaceMesh = nullptr;

public:
  DelaunaySurfaceMesh() {}
  DelaunaySurfaceMesh(lsDomainType passedLevelSet, MeshType passedMesh)
      : levelSet(passedLevelSet), surfaceMesh(passedMesh) {}
  DelaunaySurfaceMesh(psDomainType passedDomain, MeshType passedMesh)
      : levelSet(passedDomain->getLevelSets().back()), surfaceMesh(passedMesh) {
  }

  void setLevelSet(lsDomainType passedLevelSet) { levelSet = passedLevelSet; }

  void setLevelSet(psDomainType passedDomain) {
    levelSet = passedDomain->getLevelSets()->back();
  }

  void apply() {
    surfaceMesh->clear();
    const double gridDelta = levelSet->getGrid().getGridDelta();

    auto pointCloud = MeshType::New();
    viennals::ToDiskMesh<NumericType, 3>(levelSet, pointCloud).apply();

    // lsVTKWriter<NumericType>(pointCloud, "pointCloud.vtp").apply();

    vtkNew<vtkPoints> points;
    for (auto it = pointCloud->getNodes().begin();
         it != pointCloud->getNodes().end(); ++it) {
      points->InsertNextPoint((*it)[0], (*it)[1], (*it)[2]);
    }

    vtkNew<vtkPolyData> polydata;
    polydata->SetPoints(points);

    vtkNew<vtkDelaunay2D> delaunay;
    delaunay->SetInputData(polydata);
    delaunay->SetAlpha(gridDelta);
    delaunay->Update();

    auto output = delaunay->GetOutput();
    auto meshCells = output->GetPolys();
    auto meshPoints = output->GetPoints();

    auto bounds = meshPoints->GetBounds();
    int j = 0;
    for (int i = 0; i < 3; i++) {
      surfaceMesh->minimumExtent[i] = static_cast<NumericType>(bounds[j++]);
      surfaceMesh->maximumExtent[i] = static_cast<NumericType>(bounds[j++]);
    }

    double p[3];
    for (vtkIdType i = 0; i < meshPoints->GetNumberOfPoints(); i++) {
      meshPoints->GetPoint(i, p);
      surfaceMesh->insertNextNode(std::array<NumericType, 3>{
          (NumericType)p[0], (NumericType)p[1], (NumericType)p[2]});
    }

    vtkNew<vtkIdList> cellIds;
    for (vtkIdType i = 0; i < meshCells->GetNumberOfCells(); i++) {
      meshCells->GetCellAtId(i, cellIds);
      surfaceMesh->insertNextTriangle(
          std::array<unsigned, 3>{static_cast<unsigned>(cellIds->GetId(0)),
                                  static_cast<unsigned>(cellIds->GetId(1)),
                                  static_cast<unsigned>(cellIds->GetId(2))});
    }
  }
};
} // namespace viennaps