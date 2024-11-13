#pragma once

#include <lsDomain.hpp>
#include <lsToDiskMesh.hpp>
#include <vtkCellArray.h>
#include <vtkCellLocator.h>
#include <vtkDelaunay2D.h>
#include <vtkLine.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkTriangle.h>

#include <psDomain.hpp>
#include <psSmartPointer.hpp>

// WARNING: THIS ONLY WORK FOR GEOMETRIES WHICH PROJECT NICELY ON THE XY PLANE.
template <class NumericType> class culsDelaunayMesh {
  psSmartPointer<lsDomain<NumericType, 3>> levelSet = nullptr;
  psSmartPointer<lsMesh<NumericType>> surfaceMesh = nullptr;

public:
  culsDelaunayMesh() {}
  culsDelaunayMesh(psSmartPointer<lsDomain<NumericType, 3>> passedLevelSet,
                   psSmartPointer<lsMesh<NumericType>> passedMesh)
      : levelSet(passedLevelSet), surfaceMesh(passedMesh) {}
  culsDelaunayMesh(psSmartPointer<psDomain<NumericType, 3>> passedDomain,
                   psSmartPointer<lsMesh<NumericType>> passedMesh)
      : levelSet(passedDomain->getLevelSets().back()), surfaceMesh(passedMesh) {
  }

  void setLevelSet(psSmartPointer<lsDomain<NumericType, 3>> passedLevelSet) {
    levelSet = passedLevelSet;
  }

  void setLevelSet(psSmartPointer<psDomain<NumericType, 3>> passedDomain) {
    levelSet = passedDomain->getLevelSets()->back();
  }

  void apply() {
    surfaceMesh->clear();
    const double gridDelta = levelSet->getGrid().getGridDelta();

    auto pointCloud = psSmartPointer<lsMesh<NumericType>>::New();
    lsToDiskMesh<NumericType, 3>(levelSet, pointCloud).apply();

    lsVTKWriter<NumericType>(pointCloud, "pointCloud.vtp").apply();

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