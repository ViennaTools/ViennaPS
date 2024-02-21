#pragma once

#include <string>
#include <unordered_map>

#include <psUtils.hpp>

template <typename T> struct Parameters {
  // Domain
  T gridDelta = 1.0; // nm
  T xExtent = 50.;   // nm
  T yExtent = 50.;   // nm

  // Geometry
  T trenchWidth = 20.;  // nm
  T trenchHeight = 20.; // nm
  T topSpace = 10.;     // nm
  T taperAngle = 0.;    // degree

  // Process
  T processTime = 10.;
  T diffusionCoefficient = 20.0;
  T inFlux = 1.0;
  T depositionThreshold = 1.;
  T adsorptionRate = 0.1;
  T desorptionRate = 0.1;

  Parameters() {}

  void fromMap(std::unordered_map<std::string, std::string> &m) {
    psUtils::AssignItems(                                            //
        m,                                                           //
        psUtils::Item{"gridDelta", gridDelta},                       //
        psUtils::Item{"xExtent", xExtent},                           //
        psUtils::Item{"yExtent", yExtent},                           //
        psUtils::Item{"trenchWidth", trenchWidth},                   //
        psUtils::Item{"trenchHeight", trenchHeight},                 //
        psUtils::Item{"topSpace", topSpace},                         //
        psUtils::Item{"taperAngle", taperAngle},                     //
        psUtils::Item{"processTime", processTime},                   //
        psUtils::Item{"diffusionCoefficient", diffusionCoefficient}, //
        psUtils::Item{"inFlux", inFlux},                             //
        psUtils::Item{"depositionThreshold", depositionThreshold},   //
        psUtils::Item{"adsorptionRate", adsorptionRate},             //
        psUtils::Item{"desorptionRate", desorptionRate});
  }
};
