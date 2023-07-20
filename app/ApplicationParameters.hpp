#pragma once

#include <string>
#include <unordered_map>

#include <psLogger.hpp>
#include <psMaterials.hpp>
#include <psUtils.hpp>

enum class CommandType { NONE, INIT, GEOMETRY, PROCESS, OUTPUT, PLANARIZE };

enum class GeometryType { NONE, TRENCH, HOLE, PLANE, STACK, GDS, IMPORT };

enum class ProcessType {
  NONE,
  SF6O2ETCHING,
  FLUOROCARBONETCHING,
  DEPOSITION,
  SPHEREDISTRIBUTION,
  BOXDISTRIBUTION,
  DIRECTIONALETCHING,
  WETETCHING,
  ISOTROPIC
};

enum class OutputType { SURFACE, VOLUME };

#ifdef VIENNAPS_USE_DOUBLE
using NumericType = double;
#else
using NumericType = float;
#endif

struct ApplicationParameters {
  int logLevel = 2;
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
  // MAKE stack
  int numLayers = 11;
  NumericType layerHeight = 1.;
  NumericType substrateHeight = 1.;
  // GDS / IMPORT
  int layers = 0;
  std::string fileName = "";
  NumericType maskHeight = 0.1;
  int maskInvert = 0;
  NumericType xPadding = 0.;
  NumericType yPadding = 0.;
  psMaterial material = psMaterial::Si;

  // Process
  NumericType processTime = 1;
  int raysPerPoint = 3000;
  NumericType etchStopDepth = std::numeric_limits<NumericType>::lowest();
  // Plasma etching
  // fluxes in in (1e15 atoms/cm³)
  NumericType etchantFlux = 1.8e3;
  NumericType oxygenFlux = 1.0e2;
  NumericType ionFlux = 12.;
  NumericType ionEnergy = 100;     // eV
  NumericType sigmaIonEnergy = 10; // eV
  NumericType A_O = 3.;
  // Fluorocarbon etching
  NumericType temperature = 300; // K
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

  // output
  OutputType out = OutputType::SURFACE;

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
    etchantFlux = 4.5e16;
    oxygenFlux = 1e18;
    ionFlux = 2e16;
    ionEnergy = 100;
    sigmaIonEnergy = 10;
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
    numLayers = 11;
    layerHeight = 1.;
    substrateHeight = 1.;
    etchStopDepth = std::numeric_limits<NumericType>::lowest();

    if (all) {
      logLevel = 2;
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
