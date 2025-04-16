#pragma once

#include <fstream>

#include <lsBooleanOperation.hpp>
#include <lsDomain.hpp>
#include <lsFromSurfaceMesh.hpp>
#include <lsMesh.hpp>

#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

class CSVRow {
public:
  std::string operator[](std::size_t index) const {
    return std::string(&m_line[m_data[index] + 1],
                       m_data[index + 1] - (m_data[index] + 1));
  }
  std::size_t size() const { return m_data.size() - 1; }
  void readNextRow(std::istream &str) {
    std::getline(str, m_line);

    m_data.clear();
    m_data.emplace_back(-1);
    std::string::size_type pos = 0;
    while ((pos = m_line.find(',', pos)) != std::string::npos) {
      m_data.emplace_back(pos);
      ++pos;
    }
    // This checks for a trailing comma with no data after it.
    pos = m_line.size();
    m_data.emplace_back(pos);
  }

private:
  std::string m_line;
  std::vector<int> m_data;
};

std::istream &operator>>(std::istream &str, CSVRow &data) {
  data.readNextRow(str);
  return str;
}

std::ostream &operator<<(std::ostream &str, const std::array<double, 3> &arr) {
  str << "[";
  for (int i = 0; i < 2; i++) {
    str << arr[i] << ", ";
  }
  str << arr[2] << "]\n";
  return str;
}

template <class ParamsType, int D>
void ReadPath(viennaps::SmartPointer<viennaps::Domain<double, D>> domain,
              ParamsType params) {

  auto extent = params.getExtent();
  const double extent2 = extent[0] / 2.;
  double bounds[] = {-extent2, extent2, -1., 1.};
  viennals::BoundaryConditionEnum boundaryConds[] = {
      params.periodicBoundary
          ? viennals::BoundaryConditionEnum::PERIODIC_BOUNDARY
          : viennals::BoundaryConditionEnum::REFLECTIVE_BOUNDARY,
      viennals::BoundaryConditionEnum::INFINITE_BOUNDARY};

  
  std::ifstream file(params.pathFile);

  if (!file.is_open()) {
    std::cout << "Could not open path file" << std::endl;
    exit(1);
  }

  auto mesh = viennals::SmartPointer<viennals::Mesh<double>>::New();

  CSVRow row;

  double firstY, lastY;
  unsigned numPoints;
  while (file >> row) {
    std::array<double, 3> point{std::stod(row[0]) * extent2,
                                std::stod(row[1]) * extent[1] - params.offSet,
                                0.};
    numPoints = mesh->insertNextNode(point);
    if (numPoints > 0) {
      mesh->insertNextLine({numPoints, numPoints - 1});
    }
  }

  auto pointID = mesh->insertNextNode(
      {extent2 + params.gridDelta, mesh->nodes[numPoints][1]});
  mesh->insertNextLine({pointID, pointID - 1});

  pointID = mesh->insertNextNode(
      {extent2 + params.gridDelta, extent[1] + params.gridDelta});
  mesh->insertNextLine({pointID, pointID - 1});

  pointID = mesh->insertNextNode(
      {-extent2 - params.gridDelta, extent[1] + params.gridDelta});
  mesh->insertNextLine({pointID, pointID - 1});

  pointID =
      mesh->insertNextNode({-extent2 - params.gridDelta, mesh->nodes[0][1]});
  mesh->insertNextLine({pointID, pointID - 1});
  mesh->insertNextLine({0, pointID});

  // viennals::VTKWriter<double>(mesh, "test_surface.vtp").apply();

  auto cut = viennals::SmartPointer<viennals::Domain<double, D>>::New(
      bounds, boundaryConds, params.gridDelta);
  viennals::FromSurfaceMesh<double, D>(cut, mesh).apply();
  //   viennals::BooleanOperation<double, D>(cut,
  //   viennals::BooleanOperationEnum::INVERT).apply();
  domain->applyBooleanOperation(
      cut, viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT);
}