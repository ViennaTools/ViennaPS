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

  // Radical particle
  NumericType depositionRateRadical = 0.1;
  NumericType stickingProbabilityRadical = 0.1;
  NumericType reactionOrderRadical = 1.;

  // Ion particle
  NumericType depositionRateIon = 0.1;
  NumericType stickingProbabilityIon = 1e-4;
  NumericType reactionOrderIon = 1.;
  NumericType exponentIon = 100.;
  NumericType minAngleIon = 1.3962634;

  Parameters() {}

  void fromMap(std::unordered_map<std::string, std::string> &m) {
    psUtils::AssignItems(                                                        //
        m,                                                                       //
        psUtils::Item{"gridDelta", gridDelta},                                   //
        psUtils::Item{"xExtent", xExtent},                                       //
        psUtils::Item{"yExtent", yExtent},                                       //
        psUtils::Item{"trenchWidth", trenchWidth},                               //
        psUtils::Item{"trenchHeight", trenchHeight},                             //
        psUtils::Item{"taperAngle", taperAngle},                                 //
        psUtils::Item{"processTime", processTime},                               //
        psUtils::Item{"numRaysPerPoint", numRaysPerPoint},                       //
        psUtils::Item{"depositionRateRadical", depositionRateRadical},           //
        psUtils::Item{"stickingProbabilityRadical", stickingProbabilityRadical}, //
        psUtils::Item{"reactionOrderRadical", reactionOrderRadical},             //
        psUtils::Item{"depositionRateIon", depositionRateIon},                   //
        psUtils::Item{"stickingProbabilityIon", stickingProbabilityIon},         //
        psUtils::Item{"reactionOrderIon", reactionOrderIon},                     //
        psUtils::Item{"exponentIon", exponentIon},                               //
        psUtils::Item{"minAngleIon", minAngleIon}                                //
    );
  }
};
