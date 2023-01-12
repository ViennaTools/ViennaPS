#pragma once

#include <psUtils.hpp>

template <typename T> struct Parameters {
  // Domain
  T gridDelta = 0.2;
  T xExtent = 10.;
  T yExtent = 10.;

  // Geometry
  T trenchWidth = 4.;
  T trenchHeight = 8.;
  T taperAngle = 0.;

  // Process
  T processTime = 4.3;
  T sourcePower = 1.;
  T stickingProbability = 0.2;

  Parameters() {}

  void fromMap(std::unordered_map<std::string, std::string> &m) {
    psUtils::AssignItems( //
        m,                //
        // psUtils::Item{"gridDelta", gridDelta},                     //
        // psUtils::Item{"xExtent", xExtent},                         //
        // psUtils::Item{"yExtent", yExtent},                         //
        // psUtils::Item{"trenchWidth", trenchWidth},                 //
        // psUtils::Item{"trenchHeight", trenchHeight},               //
        psUtils::Item{"taperAngle", taperAngle},   //
        psUtils::Item{"processTime", processTime}, //
        // psUtils::Item{"sourcePower", sourcePower},                  //
        psUtils::Item{"stickingProbability", stickingProbability} //
    );
  }
};
