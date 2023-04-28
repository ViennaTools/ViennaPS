#pragma once

#include <string>
#include <unordered_map>

#include <psUtils.hpp>

template <typename T> struct Parameters {
  // Domain
  T gridDelta = 5.;  // nm
  T xExtent = 100.0; // nm
  T yExtent = 100.0; // nm

  // Geometry
  int numLayers = 7;
  T layerHeight = 30.;     // nm
  T substrateHeight = 50.; // nm
  T holeRadius = 75;       // nm
  T maskHeight = 50;       // nm

  // Process
  T processTime = 150;
  T totalEtchantFlux = 4.5e16;
  T totalOxygenFlux = 1e18;
  T totalIonFlux = 2e16;
  T rfBiasPower = 215; // W;

  Parameters() {}

  void fromMap(std::unordered_map<std::string, std::string> &m) {
    psUtils::AssignItems(                                    //
        m,                                                   //
        psUtils::Item{"gridDelta", gridDelta},               //
        psUtils::Item{"xExtent", xExtent},                   //
        psUtils::Item{"yExtent", yExtent},                   //
        psUtils::Item{"holeRadius", holeRadius},             //
        psUtils::Item{"numLayers", numLayers},               //
        psUtils::Item{"layerHeight", layerHeight},           //
        psUtils::Item{"substrateHeight", substrateHeight},   //
        psUtils::Item{"maskHeight", maskHeight},             //
        psUtils::Item{"totalEtchantFlux", totalEtchantFlux}, //
        psUtils::Item{"totalOxygenFlux", totalOxygenFlux},   //
        psUtils::Item{"totalIonFlux", totalIonFlux},         //
        psUtils::Item{"rfBiasPower", rfBiasPower}            //
    );
  }
};
