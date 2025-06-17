#pragma once

#include "psDomain.hpp"

#include <lsExtrude.hpp>
#include <lsToMesh.hpp>

namespace viennaps {
using namespace viennacore;

template <class NumericType> class Extrude {
  SmartPointer<Domain<NumericType, 2>> inputDomain;
  SmartPointer<Domain<NumericType, 3>> outputDomain;
  Vec2D<NumericType> extent{NumericType(0)};
  int extrudeDim = 0;
  std::array<viennals::BoundaryConditionEnum, 3> boundaryConds = {};

public:
  Extrude() = default;
  Extrude(SmartPointer<Domain<NumericType, 2>> &passedInputDomain,
          SmartPointer<Domain<NumericType, 3>> &passedOutputDomain,
          const Vec2D<NumericType> &passedExtent, const int passedExtrudeDim,
          std::array<viennals::BoundaryConditionEnum, 3> passedBoundaryConds)
      : inputDomain(passedInputDomain), outputDomain(passedOutputDomain),
        extent(passedExtent), extrudeDim(passedExtrudeDim),
        boundaryConds(passedBoundaryConds) {}

  void setInputDomain(SmartPointer<Domain<NumericType, 2>> passedInputDomain) {
    inputDomain = passedInputDomain;
  }

  // The 3D output domain will be overwritten by the extruded domain
  void
  setOutputDomain(SmartPointer<Domain<NumericType, 3>> &passedOutputDomain) {
    outputDomain = passedOutputDomain;
  }

  // Set the min and max extent in the extruded dimension
  void setExtent(const Vec2D<NumericType> &passedExtent) {
    extent = passedExtent;
  }

  // Set which index of the added dimension (x: 0, y: 1, z: 2)
  void setExtrudeDimension(const int passedExtrudeDim) {
    extrudeDim = passedExtrudeDim;
  }

  void setBoundaryConditions(
      std::array<viennals::BoundaryConditionEnum, 3> passedBoundaryConds) {
    boundaryConds = passedBoundaryConds;
  }

  void setBoundaryConditions(
      viennals::BoundaryConditionEnum passedBoundaryConds[3]) {
    for (int i = 0; i < 3; i++)
      boundaryConds[i] = passedBoundaryConds[i];
  }

  void apply() {
    if (inputDomain == nullptr) {
      Logger::getInstance()
          .addWarning("No input domain supplied to Extrude! Not converting.")
          .print();
    }
    if (outputDomain == nullptr) {
      Logger::getInstance()
          .addWarning("No output domain supplied to Extrude! Not converting.")
          .print();
      return;
    }

    auto materialMap = inputDomain->getMaterialMap();
    outputDomain->clear();

    for (std::size_t i = 0; i < inputDomain->getLevelSets().size(); i++) {
      auto tmpLS = viennals::Domain<NumericType, 3>::New();
      viennals::Extrude<NumericType>(inputDomain->getLevelSets().at(i), tmpLS,
                                     extent, extrudeDim, boundaryConds)
          .apply();

      if (Logger::getLogLevel() >= 5) {
        auto mesh = viennals::Mesh<NumericType>::New();
        viennals::ToMesh<NumericType, 3>(tmpLS, mesh).apply();
        viennals::VTKWriter<NumericType>(mesh, "extrude_layer_" +
                                                   std::to_string(i) + ".vtp")
            .apply();
      }

      if (materialMap) {
        auto material = materialMap->getMaterialAtIdx(i);
        outputDomain->insertNextLevelSetAsMaterial(tmpLS, material, false);
      } else {
        outputDomain->insertNextLevelSet(tmpLS, false);
      }
    }
  }
};
} // namespace viennaps
