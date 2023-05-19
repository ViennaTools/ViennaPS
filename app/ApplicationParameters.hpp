#pragma once

#include <string>
#include <unordered_map>

#include <psMaterials.hpp>
#include <psUtils.hpp>

enum class CommandType { NONE, INIT, GEOMETRY, PROCESS, OUTPUT, PLANARIZE };

enum class GeometryType { NONE, TRENCH, HOLE, PLANE, GDS, IMPORT };

enum class ProcessType {
  NONE,
  SF6O2ETCHING,
  DEPOSITION,
  SPHEREDISTRIBUTION,
  BOXDISTRIBUTION,
  DIRECTIONALETCHING,
  WETETCHING,
  ISOTROPIC
};

#ifdef VIENNAPS_USE_DOUBLE
using NumericType = double;
#else
using NumericType = float;
#endif

struct ApplicationParameters {
  NumericType printTimeInterval = 0;
  GeometryType geometryType = GeometryType::NONE;
  ProcessType processType = ProcessType::NONE;

  // Domain
  NumericType gridDelta = 0.02;
  NumericType xExtent = 1.0;
  NumericType yExtent = 1.0;
  int periodicBoundary = 0;
  lsIntegrationSchemeEnum integrationScheme =
      lsIntegrationSchemeEnum::ENGQUIST_OSHER_1ST_ORDER;

  // Geometry
  int mask = 0;
  int maskId = 0;
  NumericType taperAngle = 0.;
  NumericType maskZPos = 0.;
  // MAKE Trench
  NumericType trenchWidth = 0.2;
  NumericType trenchHeight = 0.5;
  // MAKE hole
  NumericType holeDepth = 0.2;
  NumericType holeRadius = 0.2;
  // GDS / IMPORT
  int layers = 0;
  std::string fileName = "";
  NumericType maskHeight = 0.1;
  int pointOrder = 0;
  int maskInvert = 0;
  NumericType xPadding = 0.;
  NumericType yPadding = 0.;
  psMaterial material = psMaterial::Si;

  // Process
  NumericType processTime = 1;
  int raysPerPoint = 3000;
  // SF6O2Etching
  NumericType totalEtchantFlux = 4.5e16;
  NumericType totalOxygenFlux = 1e18;
  NumericType totalIonFlux = 2e16;
  NumericType ionEnergy = 100;
  NumericType A_O = 3.;
  // Deposition
  NumericType rate = 1.;
  NumericType sticking = 1.;
  NumericType cosinePower = 1.;
  // Directional etching
  NumericType directionalRate = 1.;
  NumericType isotropicRate = 0.;
  std::string direction = "negZ";
  // Geometric Distributions
  // sphere
  NumericType radius = 1.;
  // box
  std::array<hrleCoordType, 3> halfAxes = {1., 1., 1.};

  ApplicationParameters() {}

  void defaultParameters(bool all = false) {
    mask = 0;
    maskId = 0;
    taperAngle = 0.;
    maskZPos = 0.;
    trenchWidth = 0.2;
    trenchHeight = 0.5;
    holeDepth = 0.2;
    holeRadius = 0.2;
    layers = 0;
    fileName = "";
    maskHeight = 0.1;
    processTime = 1;
    raysPerPoint = 3000;
    totalEtchantFlux = 4.5e16;
    totalOxygenFlux = 1e18;
    totalIonFlux = 2e16;
    ionEnergy = 100;
    A_O = 3.;
    rate = 1.;
    sticking = 1.;
    cosinePower = 1.;
    directionalRate = 1.;
    isotropicRate = 0.;
    direction = "negZ";
    radius = 1.;
    halfAxes[0] = 1.;
    halfAxes[1] = 1.;
    halfAxes[2] = 1.;

    if (all) {
      printTimeInterval = 0;
      geometryType = GeometryType::NONE;
      processType = ProcessType::NONE;
      gridDelta = 0.02;
      xExtent = 1.0;
      yExtent = 1.0;
      periodicBoundary = 0;
      integrationScheme = lsIntegrationSchemeEnum::ENGQUIST_OSHER_1ST_ORDER;
    }
  }
};
