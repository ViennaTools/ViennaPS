#pragma once

#include <string>
#include <unordered_map>

#include <psUtils.hpp>

enum class CommandType { NONE, INIT, GEOMETRY, PROCESS, OUTPUT };

enum class GeometryType { NONE, TRENCH, HOLE, PLANE, GDS, IMPORT };

enum class ProcessType {
  NONE,
  SF6O2ETCHING,
  DEPOSITION,
  GEOMETRICUNIFORMDEPOSITION
};

#ifdef VIENNAPS_USE_DOUBLE
using NumericType = double;
#else
using NumericType = float;
#endif

struct ApplicationParameters {
  int printIntermediate = 0;
  GeometryType geometryType = GeometryType::NONE;
  ProcessType processType = ProcessType::NONE;

  // Domain
  NumericType gridDelta = 0.02;
  NumericType xExtent = 1.0;
  NumericType yExtent = 1.0;
  NumericType zPos = 0.;
  int periodicBoundary = 0;

  // Geometry
  int mask = 0;
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

  ApplicationParameters() {}
};
