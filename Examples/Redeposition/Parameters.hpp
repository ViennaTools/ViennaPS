#pragma once

#include <string>
#include <unordered_map>

#include <psUtils.hpp>

template <typename T> struct Parameters {
  // Domain
  T gridDelta = 0.2;
  T xExtent = 10.0;

  // Geometry
  int numLayers = 26;
  T layerHeight = 2;
  T substrateHeight = 4;
  T holeRadius = 2;

  // Process
  T diffusionCoefficient = 10.; // diffusion cofficient
  T sink = 0.001;               // sink strength
  // convection velocity in the scallops towards the center
  T scallopStreamVelocity = 10.;
  // convection velocity in the center towards the sink on the top
  T holeStreamVelocity = 10.;

  Parameters() {}

  void fromMap(std::unordered_map<std::string, std::string> &m) {
    psUtils::AssignItems(                                              //
        m,                                                             //
        psUtils::Item{"gridDelta", gridDelta},                         //
        psUtils::Item{"xExtent", xExtent},                             //
        psUtils::Item{"numLayers", numLayers},                         //
        psUtils::Item{"layerHeight", layerHeight},                     //
        psUtils::Item{"substrateHeight", substrateHeight},             //
        psUtils::Item{"holeRadius", holeRadius},                       //
        psUtils::Item{"diffusionCoefficient", diffusionCoefficient},   //
        psUtils::Item{"sink", sink},                                   //
        psUtils::Item{"scallopStreamVelocity", scallopStreamVelocity}, //
        psUtils::Item{"holeStreamVelocity", holeStreamVelocity}        //
    );
  }
};
