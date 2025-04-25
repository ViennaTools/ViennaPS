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

#include <vcLogger.hpp>

#include <compact/psCSVReader.hpp>

template <class ParamsType, int D>
void ReadPath(viennaps::SmartPointer<viennaps::Domain<double, D>> domain,
              viennaps::SmartPointer<viennaps::Domain<double, D>> target,
              ParamsType params) {
  ReadPath(domain, params);
  ReadPath(target, params, true);
}

template <class ParamsType, int D>
void ReadPath(viennaps::SmartPointer<viennaps::Domain<double, D>> domain,
              ParamsType params, bool target = false) {

  auto extent = params.getExtent();
  double bounds[] = {0., params.halfGeometry ? extent[0] / 2. : extent[0], -1.,
                     1.};

  viennals::BoundaryConditionEnum boundaryConds[] = {
      params.periodicBoundary
          ? viennals::BoundaryConditionEnum::PERIODIC_BOUNDARY
          : viennals::BoundaryConditionEnum::REFLECTIVE_BOUNDARY,
      viennals::BoundaryConditionEnum::INFINITE_BOUNDARY};

  viennaps::CSVReader<double> reader(target ? params.targetFile
                                            : params.pathFile);
  auto dataOpt = reader.readContent();
  if (!dataOpt.has_value()) {
    viennacore::Logger::getInstance()
        .addWarning("Failed to read CSV data from " + target ? params.targetFile
                                                             : params.pathFile)
        .print();
    return;
  }

  const auto &data = dataOpt.value();
  if (data.empty()) {
    viennacore::Logger::getInstance().addWarning("CSV file is empty.").print();
    return;
  }

  auto mesh = viennals::SmartPointer<viennals::Mesh<double>>::New();
  std::vector<unsigned> nodeIDs;

  if (params.buffer > 0. && !data.empty()) {
    double x = data.front()[0] - params.gridDelta; // unshifted
    double y = data.front()[1] - params.offSet;
    nodeIDs.push_back(mesh->insertNextNode({x, y, 0.}));
  }

  for (const auto &row : data) {
    if (row.size() < 2)
      continue;
    double x = row[0] + params.buffer;
    double y = row[1] - params.offSet;
    nodeIDs.push_back(mesh->insertNextNode({x, y, 0.}));
  }

  if (params.buffer > 0. && !data.empty()) {
    double x = data.back()[0] + 2. * params.buffer + params.gridDelta;
    double y = data.back()[1] - params.offSet;
    nodeIDs.push_back(mesh->insertNextNode({x, y, 0.}));
  }

  for (std::size_t i = 1; i < nodeIDs.size(); ++i)
    mesh->insertNextLine({nodeIDs[i], nodeIDs[i - 1]});

  auto cut = viennals::SmartPointer<viennals::Domain<double, D>>::New(
      bounds, boundaryConds, params.gridDelta);
  viennals::FromSurfaceMesh<double, D>(cut, mesh).apply();
  domain->applyBooleanOperation(
      cut, viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT);
}