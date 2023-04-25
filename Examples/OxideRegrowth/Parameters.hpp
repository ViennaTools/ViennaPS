#pragma once

#include <string>
#include <unordered_map>

#include <psUtils.hpp>

template <typename T> struct Parameters {
  // Domain
  // all length units in nm
  T etchRate = 6.5; // nm/min
  T oxideEtchRate = 0.;

  T gridDelta = 2.;  // nm
  T xExtent = 300.0; // nm

  // Geometry
  int numLayers = 7;
  T layerHeight = 30.;     // nm
  T substrateHeight = 50.; // nm
  T trenchWidth = 150.;    // nm

  // Process
  T targetEtchDepth = 200.;     // nm
  T diffusionCoefficient = 50.; // diffusion cofficient nm²/s
  T sink = 0.01;                // sink strength
  // convection velocity in the scallops towards the center nm/s
  T scallopStreamVelocity = 5.;
  // convection velocity in the center towards the sink on the top nm/s
  T holeStreamVelocity = 5.;
  T redepoFactor = 0.7;
  T redepoThreshold = 0.2;
  T redepoTimeInt = 60.;

  Parameters() {}

  void fromMap(std::unordered_map<std::string, std::string> &m) {
    psUtils::AssignItems(                                              //
        m,                                                             //
        psUtils::Item{"gridDelta", gridDelta},                         //
        psUtils::Item{"xExtent", xExtent},                             //
        psUtils::Item{"numLayers", numLayers},                         //
        psUtils::Item{"layerHeight", layerHeight},                     //
        psUtils::Item{"substrateHeight", substrateHeight},             //
        psUtils::Item{"trenchWidth", trenchWidth},                     //
        psUtils::Item{"diffusionCoefficient", diffusionCoefficient},   //
        psUtils::Item{"sink", sink},                                   //
        psUtils::Item{"scallopStreamVelocity", scallopStreamVelocity}, //
        psUtils::Item{"holeStreamVelocity", holeStreamVelocity},       //
        psUtils::Item{"targetEtchDepth", targetEtchDepth},             //
        psUtils::Item{"redepoFactor", redepoFactor},                   //
        psUtils::Item{"redepoThreshold", redepoThreshold},             //
        psUtils::Item{"redepoTimeInt", redepoTimeInt},                 //
        psUtils::Item{"oxideEtchRate", oxideEtchRate}                  //
    );
  }
};
