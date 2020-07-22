#ifndef CS_VTK_WRITER_HPP
#define CS_VTK_WRITER_HPP

#include <vtkDoubleArray.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkRectilinearGrid.h>
#include <vtkSmartPointer.h>
#include <vtkXMLPolyDataWriter.h>
#include <vtkXMLRectilinearGridWriter.h>

#include <hrleDenseIterator.hpp>
#include <lsMessage.hpp>

#include <lsMesh.hpp>
#include <lsVTKWriter.hpp>

#include <csDomain.hpp>

template <class T, int D> class csVTKWriter {
  using csDomainType = lsSmartPointer<const csDomain<T, D>>;
  csDomainType cellSet = nullptr;
  std::string fileName;

public:
  csVTKWriter() {}

  csVTKWriter(csDomainType passedCellSet) : cellSet(&passedCellSet) {}

  csVTKWriter(csDomainType passedCellSet, std::string passedFileName)
      : cellSet(passedCellSet), fileName(passedFileName) {}

  void setCellSet(csDomainType passedCellSet) { cellSet = &passedCellSet; }

  void setFileName(std::string passedFileName) { fileName = passedFileName; }

  void writeVTP() {
    if (cellSet == nullptr) {
      lsMessage::getInstance()
          .addWarning("No cellSet was passed to csVTKWriter. Not writing.")
          .print();
      return;
    }
    // check filename
    if (fileName.empty()) {
      lsMessage::getInstance()
          .addWarning("No file name specified for csVTKWriter. Not writing.")
          .print();
      return;
    }

    if (cellSet->getNumberOfCells() == 0) {
      lsMessage::getInstance()
          .addWarning("CellSet passed to csVTKWriter is empty. Not writing.")
          .print();
      return;
    }

    lsMesh mesh;
    auto gridDelta = cellSet->getGrid().getGridDelta();

    vtkSmartPointer<vtkPoints> polyPoints = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkCellArray> polyCells =
        vtkSmartPointer<vtkCellArray>::New();

    std::map<unsigned, vtkSmartPointer<vtkFloatArray>> pointDataMap;

    unsigned counter = 0;
    for (hrleConstSparseIterator<
             typename csDomainType::element_type::DomainType>
             it(cellSet->getDomain());
         !it.isFinished(); ++it) {
      if (!it.isDefined())
        continue;

      auto index = it.getStartIndices();

      hrleVectorType<double, 3> point(double(0));
      for (unsigned i = 0; i < D; ++i) {
        point[i] = index[i] * gridDelta;
      }

      // Points
      polyPoints->InsertNextPoint(point[0], point[1], point[2]);

      // insert vertex
      polyCells->InsertNextCell(1);
      polyCells->InsertCellPoint(counter);

      // insert material fraction to correct pointData value
      auto &materialFractions = it.getValue().getMaterialFractions();
      // try if each material of the cell already exists
      for (auto &material : materialFractions) {
        auto it = pointDataMap.find(material.first);
        // if material array does not exist yet
        if (it == pointDataMap.end()) {
          auto pointData = vtkSmartPointer<vtkFloatArray>::New();
          pointData->SetNumberOfComponents(1);
          pointData->SetName(
              ("Material " + std::to_string(material.first)).c_str());
          // fill up until current index
          for (unsigned i = 0; i < counter; ++i) {
            pointData->InsertNextValue(material.first);
          }
          // now insert into map
          it = pointDataMap.insert(std::make_pair(material.first, pointData))
                   .first;
        }

        it->second->InsertNextValue(material.second);
      }

      ++counter;
    }

    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
    polyData->SetPoints(polyPoints);
    polyData->SetVerts(polyCells);
    for (auto it : pointDataMap) {
      polyData->GetCellData()->AddArray(it.second);
    }

    vtkSmartPointer<vtkXMLPolyDataWriter> pwriter =
        vtkSmartPointer<vtkXMLPolyDataWriter>::New();
    pwriter->SetFileName(fileName.c_str());
    pwriter->SetInputData(polyData);
    pwriter->Write();
  }

  void apply() {
    if (cellSet == nullptr) {
      lsMessage::getInstance()
          .addWarning("No cellSet was passed to csVTKWriter. Not writing.")
          .print();
      return;
    }
    // check filename
    if (fileName.empty()) {
      lsMessage::getInstance()
          .addWarning("No file name specified for csVTKWriter. Not writing.")
          .print();
      return;
    }

    if (cellSet->getNumberOfCells() == 0) {
      lsMessage::getInstance()
          .addWarning("CellSet passed to csVTKWriter is empty. Not writing.")
          .print();
      return;
    }

    // convert csDomain into vtkImageData
    // vtkSmartPointer<vtkImageData> imageData =
    // vtkSmartPointer<vtkImageData>::New(); imageData->SetDimensions(extent[0],
    // extent[1], extent[2]); imageData->SetOrigin(min[0], min[1], min[2]);
    // imageData->SetSpacing(gridDelta, gridDelta, gridDelta);
    // imageData->SetScalarTypeToDouble();
    // imageData->AllocateScalars();

    auto &grid = cellSet->getGrid();
    auto &domain = cellSet->getDomain();
    auto gridDelta = grid.getGridDelta();

    // convert csDomain into vtkRectilinearGrid
    vtkSmartPointer<vtkDoubleArray>
        coords[3]; // needs to be 3 because vtk only knows 3D

    // fill grid with points
    for (unsigned i = 0; i < D; ++i) {
      int gridMin = 0, gridMax = 0;
      coords[i] = vtkSmartPointer<vtkDoubleArray>::New();

      if (grid.getBoundaryConditions(i) ==
          csDomainType::element_type::GridType::boundaryType::
              INFINITE_BOUNDARY) {
        gridMin = domain.getMinRunBreak(i);
        gridMax = domain.getMaxRunBreak(i);
      } else {
        gridMin = grid.getMinBounds(i);
        gridMax = grid.getMaxBounds(i) + 1;
      }

      for (int x = gridMin; x <= gridMax; ++x) {
        coords[i]->InsertNextValue((double(x) - 0.5) * gridDelta);
      }
    }

    // in 2D, just add 1 grid point at origin
    if (D == 2) {
      coords[2] = vtkSmartPointer<vtkDoubleArray>::New();
      coords[2]->InsertNextValue(0);
    }

    vtkSmartPointer<vtkRectilinearGrid> rgrid =
        vtkSmartPointer<vtkRectilinearGrid>::New();
    rgrid->SetDimensions(coords[0]->GetNumberOfTuples(),
                         coords[1]->GetNumberOfTuples(),
                         coords[2]->GetNumberOfTuples());
    rgrid->SetXCoordinates(coords[0]);
    rgrid->SetYCoordinates(coords[1]);
    rgrid->SetZCoordinates(coords[2]);

    // Make array to store filling fractions
    std::map<unsigned, vtkSmartPointer<vtkFloatArray>> pointDataMap;
    // vtkSmartPointer<vtkFloatArray> fillingFractions =
    //     vtkSmartPointer<vtkFloatArray>::New();
    // fillingFractions->SetNumberOfComponents(1);
    // fillingFractions->SetName("FillingFractions");
    // pointDataMap.insert(std::make_pair(0, fillingFractions));

    vtkIdType pointId = 0;
    // int yValue = -10000000;
    // std::cout << std::endl << "RECTILINEAR GRID" << std::endl;
    // std::cout << gridMinima << " - " << gridMaxima << std::endl;
    for (hrleConstDenseIterator<typename csDomainType::element_type::DomainType>
             it(domain);
         !it.isFinished(); it.next()) {

      // TODO DENSE ITERATOR IS BROKEN, IT SKIPS ONE POINT FOR EACH NEW LINE
      // double p[3];
      // rgrid->GetPoint(pointId, p);
      // hrleVectorType<float, D> point(p);
      // point /= gridDelta;
      // std::cout << "grid: " << point << ", cs: " << it.getIndices() <<
      // std::endl;
      //
      // while(point < it.getStartIndices()){
      //   rgrid->GetPoint(pointId, p);
      //   point = hrleVectorType<float, D>(p);
      //   fillingFractions->InsertNextValue(it.getValue().getFillingFraction());
      //   std::cout << it.getStartIndices() << ": " <<
      //   it.getValue().getFillingFraction() << std::endl;
      //   ++pointId;
      // }
      // fillingFractions->InsertNextValue(it.getValue().getFillingFraction());
      // std::cout << it.getIndices() << ": " <<
      // it.getValue().getFillingFraction() << std::endl; std::cout <<
      // it.getIndices() << ": " << it.getIteratorIndices() << std::endl;
      // if(it.getIndices()[1] != yValue) {
      //   yValue = it.getIndices()[1];
      //   std::cout << std::endl;
      // }
      // std::cout << std::setw(8) << it.getValue() << "  ";

      // pointDataMap.find(0)->second->InsertNextValue(0);
      // insert material fraction to correct pointData value
      auto materialFractions = it.getValue().getMaterialFractions();
      // try if each material of the cell already exists
      // if(materialFractions.empty()) {
      //   materialFractions.push_back(std::make_pair(0, 0));
      // }

      for (auto &material : materialFractions) {
        auto it = pointDataMap.find(material.first);
        // if material array does not exist yet
        if (it == pointDataMap.end()) {
          auto pointData = vtkSmartPointer<vtkFloatArray>::New();
          pointData->SetNumberOfComponents(1);
          pointData->SetName(
              ("Material " + std::to_string(material.first)).c_str());
          // fill up until current index
          for (unsigned i = 0; i < pointId; ++i) {
            pointData->InsertNextValue(0);
          }
          // now insert into map
          it = pointDataMap.insert(std::make_pair(material.first, pointData))
                   .first;
        }

        it->second->InsertNextValue(material.second);
      }

      ++pointId;
      if (pointId >= rgrid->GetNumberOfPoints())
        break;
    }

    // rgrid->GetPointData()->SetScalars(fillingFractions);
    for (auto it : pointDataMap) {
      rgrid->GetCellData()->AddArray(it.second);
    }

    vtkSmartPointer<vtkXMLRectilinearGridWriter> gwriter =
        vtkSmartPointer<vtkXMLRectilinearGridWriter>::New();
    gwriter->SetFileName(fileName.c_str());
    gwriter->SetInputData(rgrid);
    gwriter->Write();
  }
};

#endif // CS_VTK_WRITER_HPP
