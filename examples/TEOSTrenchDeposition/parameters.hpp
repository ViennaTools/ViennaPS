#pragma once

#include <string>
#include <unordered_map>

#include <psUtils.hpp>

template <class NumericType> struct Parameters {
  // Domain
  NumericType gridDelta = 5.; // um
  NumericType xExtent = 100.; // um
  NumericType yExtent = 100.; // um (3D mode only)

  // Geometry
  NumericType trenchWidth = 70;  // um
  NumericType trenchHeight = 70; // um
  NumericType taperAngle = 0.;   // degrees

  // Process
  NumericType processTime = 350.; // min
  int numRaysPerPoint = 2000;

  // 1. particle
  NumericType depositionRateP1 = 0.1;
  NumericType stickingProbabilityP1 = 0.1;
  NumericType reactionOrderP1 = 1.;

  // 2. particle
  NumericType depositionRateP2 = 0.01;
  NumericType stickingProbabilityP2 = 1e-4;
  NumericType reactionOrderP2 = 1.;

  Parameters() {}

  void fromMap(std::unordered_map<std::string, std::string> &m) {
    psUtils::AssignItems(                                              //
        m,                                                             //
        psUtils::Item{"gridDelta", gridDelta},                         //
        psUtils::Item{"xExtent", xExtent},                             //
        psUtils::Item{"yExtent", yExtent},                             //
        psUtils::Item{"trenchWidth", trenchWidth},                     //
        psUtils::Item{"trenchHeight", trenchHeight},                   //
        psUtils::Item{"taperAngle", taperAngle},                       //
        psUtils::Item{"processTime", processTime},                     //
        psUtils::Item{"numRaysPerPoint", numRaysPerPoint},             //
        psUtils::Item{"depositionRateP1", depositionRateP1},           //
        psUtils::Item{"stickingProbabilityP1", stickingProbabilityP1}, //
        psUtils::Item{"reactionOrderP1", reactionOrderP1},             //
        psUtils::Item{"depositionRateP2", depositionRateP2},           //
        psUtils::Item{"stickingProbabilityP2", stickingProbabilityP2}, //
        psUtils::Item{"reactionOrderP2", reactionOrderP2}              //
    );
  }
};
