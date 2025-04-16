#pragma once

#include <lsMakeGeometry.hpp>
#include <lsToDiskMesh.hpp>
#include <lsToSurfaceMesh.hpp>
#include <psDomain.hpp>
#include <psPlanarize.hpp>

template <int D>
void printLS(viennals::SmartPointer<viennals::Domain<double, D>> domain,
             std::string fileName) {
  auto mesh = viennals::SmartPointer<viennals::Mesh<double>>::New();
  viennals::ToSurfaceMesh<double, D>(domain, mesh).apply();
  viennals::VTKWriter<double>(mesh, fileName).apply();
}

template <class ParamsType>
std::pair<double, double>
MeasureDepth(viennaps::SmartPointer<viennaps::Domain<double, 2>> domain,
             ParamsType params) {
  std::pair<double, double> depth{-1., -1.};

  auto copyDomain = viennaps::SmartPointer<viennaps::Domain<double, 2>>::New();
  copyDomain->deepCopy(domain);

  const double pillarMaxPos = params.numPillars == 1
                                  ? std::numeric_limits<double>::max()
                                  : params.pillarMaxPos;

  viennaps::Planarize<double, 2>(copyDomain, params.measureHeight).apply();
  auto mesh = viennaps::SmartPointer<viennals::Mesh<double>>::New();
  viennals::ToDiskMesh<double, 2>(copyDomain->getLevelSets().back(), mesh)
      .apply();
  double maxYPos = std::numeric_limits<double>::lowest();

  // measure first pillar
  for (const auto node : mesh->nodes) {
    if (node[1] > maxYPos && node[0] < pillarMaxPos)
      maxYPos = node[1];
  }

  std::vector<std::array<double, 3>> topNodes;
  for (const auto node : mesh->nodes) {
    if (node[1] > maxYPos - params.gridDelta / 2. && node[0] < pillarMaxPos)
      topNodes.push_back(node);
  }

  double xPos_out_SiGe = topNodes[0][0];
  double xPos_in_SiGe = topNodes[0][0];

  for (const auto node : topNodes) {
    if (node[0] < xPos_out_SiGe)
      xPos_out_SiGe = node[0];

    if (node[0] > xPos_in_SiGe)
      xPos_in_SiGe = node[0];
  }

  viennaps::Planarize<double, 2>(copyDomain, params.measureHeight -
                                                 params.measureHeightDelta)
      .apply();
  viennals::ToDiskMesh<double, 2>(copyDomain->getLevelSets().back(), mesh)
      .apply();

  maxYPos = std::numeric_limits<double>::lowest();
  for (const auto node : mesh->nodes) {
    if (node[1] > maxYPos && node[0] < pillarMaxPos)
      maxYPos = node[1];
  }

  topNodes.clear();
  for (const auto node : mesh->nodes) {
    if (node[1] > maxYPos - params.gridDelta / 2. && node[0] < pillarMaxPos)
      topNodes.push_back(node);
  }

  double xPos_out_Si = topNodes[0][0];
  double xPos_in_Si = topNodes[0][0];

  for (const auto node : topNodes) {
    if (node[0] < xPos_out_Si)
      xPos_out_Si = node[0];

    if (node[0] > xPos_in_Si)
      xPos_in_Si = node[0];
  }

  depth.first =
      std::abs(xPos_out_SiGe - xPos_out_Si); // depth on left side of pillar
  depth.second =
      std::abs(xPos_in_Si - xPos_in_SiGe); // depth on right side of pillar

  return depth;
}

template <class ParamsType, int D>
void MakeInitialGeometry(
    viennaps::SmartPointer<viennaps::Domain<double, D>> domain,
    ParamsType params) {
  const double totalHeight = params.numLayers * params.layerHeight;
  auto extent = params.getExtent();
  double bounds[] = {-extent[0] / 2., params.halfGeometry ? 0 : extent[0] / 2.,
                     -1., 1.};
  viennals::BoundaryConditionEnum boundaryConds[] = {
      params.periodicBoundary
          ? viennals::BoundaryConditionEnum::PERIODIC_BOUNDARY
          : viennals::BoundaryConditionEnum::REFLECTIVE_BOUNDARY,
      viennals::BoundaryConditionEnum::INFINITE_BOUNDARY};

  double normal[D] = {0., 1.};
  double origin[D] = {0., 0.};

  // substrate plane
  {
    auto plane = viennals::SmartPointer<viennals::Domain<double, D>>::New(
        bounds, boundaryConds, params.gridDelta);
    viennals::MakeGeometry<double, D>(
        plane,
        viennals::SmartPointer<viennals::Plane<double, D>>::New(origin, normal))
        .apply();
    domain->insertNextLevelSetAsMaterial(plane, viennaps::Material::Si);
  }

  // alternating layers
  for (int i = 0; i < params.numLayers; i++) {
    origin[1] += params.layerHeight;
    auto plane = viennals::SmartPointer<viennals::Domain<double, D>>::New(
        bounds, boundaryConds, params.gridDelta);
    viennals::MakeGeometry<double, D>(
        plane,
        viennals::SmartPointer<viennals::Plane<double, D>>::New(origin, normal))
        .apply();
    if (i % 2 == 0) {
      domain->insertNextLevelSetAsMaterial(plane, viennaps::Material::SiGe);
    } else {
      domain->insertNextLevelSetAsMaterial(plane, viennaps::Material::Si);
    }
  }

  // SiO2 mask
  {
    double maskPosY = totalHeight + params.maskHeight;
    origin[1] = maskPosY;
    auto plane = viennals::SmartPointer<viennals::Domain<double, D>>::New(
        bounds, boundaryConds, params.gridDelta);
    viennals::MakeGeometry<double, D>(
        plane,
        viennals::SmartPointer<viennals::Plane<double, D>>::New(origin, normal))
        .apply();
    domain->insertNextLevelSetAsMaterial(plane, viennaps::Material::SiO2);
  }

  if (!params.pathFile.empty())
    return;

  // mask
  {
    double maskPosY = totalHeight + params.maskHeight + 5 * params.gridDelta;
    origin[1] = maskPosY;
    auto plane = viennals::SmartPointer<viennals::Domain<double, D>>::New(
        bounds, boundaryConds, params.gridDelta);
    viennals::MakeGeometry<double, D>(
        plane,
        viennals::SmartPointer<viennals::Plane<double, D>>::New(origin, normal))
        .apply();
    domain->insertNextLevelSetAsMaterial(plane, viennaps::Material::Mask);

    // left right space
    double minPoint[] = {-extent[0] / 2. - params.gridDelta,
                         totalHeight + params.maskHeight};
    double maxPoint[] = {-extent[0] / 2. + params.lateralSpacing,
                         maskPosY + params.gridDelta};

    auto box = viennals::SmartPointer<viennals::Domain<double, D>>::New(
        bounds, boundaryConds, params.gridDelta);
    viennals::MakeGeometry<double, D>(
        box, viennals::SmartPointer<viennals::Box<double, D>>::New(minPoint,
                                                                   maxPoint))
        .apply();

    domain->applyBooleanOperation(
        box, viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT);

    if (params.halfGeometry == 0) {
      minPoint[0] = extent[0] / 2. - params.lateralSpacing;
      maxPoint[0] = extent[0] / 2. + params.gridDelta;

      viennals::MakeGeometry<double, D>(
          box, viennals::SmartPointer<viennals::Box<double, D>>::New(minPoint,
                                                                     maxPoint))
          .apply();
      domain->applyBooleanOperation(
          box, viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT);
    }

    double xpos = -extent[0] / 2. + params.lateralSpacing + params.maskWidth;
    int pillars =
        params.halfGeometry ? params.numPillars / 2 : params.numPillars;
    for (int i = 0; i < pillars; i++) {
      minPoint[0] = xpos;
      maxPoint[0] = xpos + params.trenchWidth;

      viennals::MakeGeometry<double, D>(
          box, viennals::SmartPointer<viennals::Box<double, D>>::New(minPoint,
                                                                     maxPoint))
          .apply();
      domain->applyBooleanOperation(
          box, viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT);

      xpos += params.maskWidth + params.trenchWidth;
    }
  }
}