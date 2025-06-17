#pragma once

#include "../psDomain.hpp"

#include <lsBooleanOperation.hpp>
#include <lsFromSurfaceMesh.hpp>
#include <lsMakeGeometry.hpp>
#include <lsToSurfaceMesh.hpp>

#include <cmath>

namespace viennaps {

using namespace viennacore;

template <class NumericType, int D> class GeometryFactory {
protected:
  using lsDomainType = SmartPointer<viennals::Domain<NumericType, D>>;

  const DomainSetup<NumericType, D> &setup_;
  const std::string name_;

public:
  GeometryFactory(const DomainSetup<NumericType, D> &domainSetup,
                  const std::string &name = "GeometryFactory")
      : setup_(domainSetup), name_(name) {}

  lsDomainType makeSubstrate(NumericType base) {

    auto substrate = lsDomainType::New(setup_.grid());

    NumericType normal[D] = {0.};
    NumericType origin[D] = {0.};

    normal[D - 1] = 1.;
    origin[D - 1] = base;
    viennals::MakeGeometry<NumericType, D>(
        substrate, viennals::Plane<NumericType, D>::New(origin, normal))
        .apply();

    if (Logger::getInstance().getLogLevel() >= 5) {
      saveSurfaceMesh(substrate, "_Substrate");
    }

    return substrate;
  }

  lsDomainType makeMask(NumericType base, NumericType height) {
    assert(setup_.isValid());

    auto mask = lsDomainType::New(setup_.grid());

    NumericType normal[D] = {0.};
    NumericType origin[D] = {0.};

    normal[D - 1] = 1.;
    origin[D - 1] = base + height;
    viennals::MakeGeometry<NumericType, D>(
        mask, viennals::Plane<NumericType, D>::New(origin, normal))
        .apply();

    auto maskAdd = lsDomainType::New(setup_.grid());
    origin[D - 1] = base;
    normal[D - 1] = -1.;
    viennals::MakeGeometry<NumericType, D>(
        maskAdd, viennals::Plane<NumericType, D>::New(origin, normal))
        .apply();

    viennals::BooleanOperation<NumericType, D>(
        mask, maskAdd, viennals::BooleanOperationEnum::INTERSECT)
        .apply();

    if (Logger::getInstance().getLogLevel() >= 5) {
      saveSurfaceMesh(mask, "_Mask");
    }

    return mask;
  }

  lsDomainType makeCylinderStencil(std::array<NumericType, D> position,
                                   NumericType radius, NumericType height,
                                   NumericType angle) {
    assert(setup_.isValid());

    auto cutout = lsDomainType::New(setup_.grid());

    NumericType normal[D] = {0.};
    normal[D - 1] = 1.;

    NumericType topRadius = radius + std::tan(angle * M_PI / 180.) * height;
    viennals::MakeGeometry<NumericType, D>(
        cutout, viennals::Cylinder<NumericType, D>::New(
                    position.data(), normal, height, radius, topRadius))
        .apply();

    if (Logger::getInstance().getLogLevel() >= 5) {
      static int count = 0;
      saveSurfaceMesh(cutout, "_Cylinder" + std::to_string(count++));
    }

    return cutout;
  }

  lsDomainType makeBoxStencil(std::array<NumericType, D> position,
                              NumericType width, NumericType height,
                              NumericType angle) {
    if (angle >= 90 || angle <= -90) {
      Logger::getInstance()
          .addError(name_ +
                    ": Taper angle must be between -90 and 90 "
                    "degrees: " +
                    std::to_string(angle))
          .print();
    }

    auto cutout = lsDomainType::New(setup_.grid());
    auto gridDelta = setup_.gridDelta();
    auto yExt = setup_.yExtent() / 2 + gridDelta;

    auto mesh = viennals::Mesh<NumericType>::New();
    const NumericType offSet = height * std::tan(angle * M_PI / 180.);

    if constexpr (D == 2) {
      auto const x = position[0];
      auto const y = position[1];
      mesh->insertNextNode(Vec3D<NumericType>{x - width / 2, y, 0.});
      mesh->insertNextNode(Vec3D<NumericType>{x + width / 2, y, 0.});
      mesh->insertNextLine({1, 0});

      if (offSet >= width / 2) { // single top node
        NumericType top = y + width * height / (2 * offSet);
        mesh->insertNextNode(Vec3D<NumericType>{x, top, 0.});
        mesh->insertNextLine({2, 1});
        mesh->insertNextLine({0, 2});
      } else {
        mesh->insertNextNode(
            Vec3D<NumericType>{x + width / 2 - offSet, y + height, 0.});
        mesh->insertNextNode(
            Vec3D<NumericType>{x - width / 2 + offSet, y + height, 0.});
        mesh->insertNextLine({2, 1});
        mesh->insertNextLine({3, 2});
        mesh->insertNextLine({0, 3});
      }
    } else { // 3D
      auto const x = position[0];
      auto const base = position[2];
      mesh->insertNextNode(Vec3D<NumericType>{x - width / 2, yExt, base});
      mesh->insertNextNode(Vec3D<NumericType>{x + width / 2, yExt, base});

      if (offSet >= width / 2) { // single top node
        NumericType top = base + width * height / (2 * offSet);
        mesh->insertNextNode(Vec3D<NumericType>{x, yExt, top});

        // shifted nodes by y extent
        mesh->insertNextNode(Vec3D<NumericType>{x - width / 2, -yExt, base});
        mesh->insertNextNode(Vec3D<NumericType>{x + width / 2, -yExt, base});
        mesh->insertNextNode(Vec3D<NumericType>{x, -yExt, top});

        // triangles
        mesh->insertNextTriangle({0, 2, 1}); // front
        mesh->insertNextTriangle({3, 4, 5}); // back
        mesh->insertNextTriangle({0, 1, 3}); // bottom
        mesh->insertNextTriangle({1, 4, 3}); // bottom
        mesh->insertNextTriangle({1, 2, 5}); // right
        mesh->insertNextTriangle({1, 5, 4}); // right
        mesh->insertNextTriangle({0, 3, 2}); // left
        mesh->insertNextTriangle({3, 5, 2}); // left
      } else {
        mesh->insertNextNode(
            Vec3D<NumericType>{x + width / 2 - offSet, yExt, base + height});
        mesh->insertNextNode(
            Vec3D<NumericType>{x - width / 2 + offSet, yExt, base + height});

        // shifted nodes by y extent
        mesh->insertNextNode(Vec3D<NumericType>{x - width / 2, -yExt, base});
        mesh->insertNextNode(Vec3D<NumericType>{x + width / 2, -yExt, base});
        mesh->insertNextNode(
            Vec3D<NumericType>{x + width / 2 - offSet, -yExt, base + height});
        mesh->insertNextNode(
            Vec3D<NumericType>{x - width / 2 + offSet, -yExt, base + height});

        // triangles
        mesh->insertNextTriangle({0, 3, 1}); // front
        mesh->insertNextTriangle({1, 3, 2}); // front
        mesh->insertNextTriangle({4, 5, 6}); // back
        mesh->insertNextTriangle({4, 6, 7}); // back
        mesh->insertNextTriangle({0, 1, 4}); // bottom
        mesh->insertNextTriangle({1, 5, 4}); // bottom
        mesh->insertNextTriangle({1, 2, 5}); // right
        mesh->insertNextTriangle({2, 6, 5}); // right
        mesh->insertNextTriangle({0, 4, 3}); // left
        mesh->insertNextTriangle({3, 4, 7}); // left
        mesh->insertNextTriangle({3, 7, 2}); // top
        mesh->insertNextTriangle({2, 7, 6}); // top
      }
    }
    viennals::FromSurfaceMesh<NumericType, D>(cutout, mesh).apply();

    if (Logger::getInstance().getLogLevel() >= 5) {
      static int count = 0;
      saveSurfaceMesh(cutout, "_Trench" + std::to_string(count++));
    }

    return cutout;
  }

private:
  void saveSurfaceMesh(lsDomainType levelSet, const std::string &name) {
    auto mesh = viennals::Mesh<NumericType>::New();
    viennals::ToSurfaceMesh<NumericType, D>(levelSet, mesh).apply();
    viennals::VTKWriter<NumericType>(mesh, name_ + name).apply();
  }
};

} // namespace viennaps
