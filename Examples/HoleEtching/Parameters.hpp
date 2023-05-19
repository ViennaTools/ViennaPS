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
  T processTime = 150; // s
  T totalIonFlux = 12.;
  T totalEtchantFlux = 1.8e3;
  T totalOxygenFlux = 1.0e2;
  T rfBias = 50; // W
  T A_O = 2.;

  int raysPerPoint = 1000;

  Parameters() {}

  void fromMap(std::unordered_map<std::string, std::string> &m) {
    psUtils::AssignItems(                                    //
        m,                                                   //
        psUtils::Item{"gridDelta", gridDelta},               //
        psUtils::Item{"xExtent", xExtent},                   //
        psUtils::Item{"yExtent", yExtent},                   //
        psUtils::Item{"holeRadius", holeRadius},             //
        psUtils::Item{"maskHeight", maskHeight},             //
        psUtils::Item{"taperAngle", taperAngle},             //
        psUtils::Item{"totalEtchantFlux", totalEtchantFlux}, //
        psUtils::Item{"totalOxygenFlux", totalOxygenFlux},   //
        psUtils::Item{"totalIonFlux", totalIonFlux},         //
        psUtils::Item{"rfBias", rfBias},                     //
        psUtils::Item{"A_O", A_O},                           //
        psUtils::Item{"raysPerPoint", raysPerPoint}          //
    );
  }
};
