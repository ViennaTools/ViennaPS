#pragma once

#include <string>
#include <unordered_map>

#include <psUtils.hpp>

template <typename T> struct Parameters {
  // Domain
  T gridDelta = 0.05; // um
  T xExtent = 1.0;    // um
  T yExtent = 1.0;    // um

  // Geometry
  T holeRadius = 0.2; // um
  T maskHeight = 0.2; // um
  T taperAngle = 0.;  // degree

  // Process
  T processTime = 100; // s
  T ionFlux = 12.;
  T etchantFlux = 1.8e3;
  T oxygenFlux = 1.0e2;
  T meanEnergy = 100; // eV
  T sigmaEnergy = 10; // eV
  T A_O = 2.;

  T etchStopDepth = -100;

  int raysPerPoint = 1000;

  Parameters() {}

  void fromMap(std::unordered_map<std::string, std::string> &m) {
    psUtils::AssignItems(                              //
        m,                                             //
        psUtils::Item{"gridDelta", gridDelta},         //
        psUtils::Item{"xExtent", xExtent},             //
        psUtils::Item{"yExtent", yExtent},             //
        psUtils::Item{"holeRadius", holeRadius},       //
        psUtils::Item{"maskHeight", maskHeight},       //
        psUtils::Item{"taperAngle", taperAngle},       //
        psUtils::Item{"processTime", processTime},     //
        psUtils::Item{"etchantFlux", etchantFlux},     //
        psUtils::Item{"oxygenFlux", oxygenFlux},       //
        psUtils::Item{"ionFlux", ionFlux},             //
        psUtils::Item{"meanEnergy", meanEnergy},       //
        psUtils::Item{"sigmaEnergy", sigmaEnergy},     //
        psUtils::Item{"A_O", A_O},                     //
        psUtils::Item{"etchStopDepth", etchStopDepth}, //
        psUtils::Item{"raysPerPoint", raysPerPoint}    //
    );
  }
};
