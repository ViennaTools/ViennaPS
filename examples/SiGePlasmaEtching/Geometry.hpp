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

template <class ParamsType, int D>
void MakeInitialGeometry(
    viennaps::SmartPointer<viennaps::Domain<double, D>> domain,
    ParamsType params) {
  const double totalHeight = params.numLayers * params.layerHeight;

  auto extent = params.getExtent();

  double bounds[] = {0., params.halfGeometry ? extent[0] / 2. : extent[0], -1.,
                     1.};
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