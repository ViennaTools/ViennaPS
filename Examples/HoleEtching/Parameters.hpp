#pragma once

#include <string>
#include <unordered_map>

#include <psUtils.hpp>

template <typename T> struct Parameters {
  // Domain
  T gridDelta = 0.02;
  T xExtent = 1.0;
  T yExtent = 1.0;

  // Geometry
  T holeRadius = 0.2;
  T topRadius = 0.2;
  T maskHeight = 0.2;
  T taperAngle = 0.;

  // Process
  T processTime = 150;
  T totalEtchantFlux = 4.5e16;
  T totalOxygenFlux = 1e18;
  T totalIonFlux = 2e16;
  T ionEnergy = 100;
  T A_O = 3.;

  Parameters() {}

  void fromMap(std::unordered_map<std::string, std::string> &m) {
    psUtils::AssignItems(                                    //
        m,                                                   //
        psUtils::Item{"gridDelta", gridDelta},               //
        psUtils::Item{"xExtent", xExtent},                   //
        psUtils::Item{"yExtent", yExtent},                   //
        psUtils::Item{"holeRadius", holeRadius},             //
        psUtils::Item{"topRadius", topRadius},               //
        psUtils::Item{"maskHeight", maskHeight},             //
        psUtils::Item{"taperAngle", taperAngle},             //
        psUtils::Item{"totalEtchantFlux", totalEtchantFlux}, //
        psUtils::Item{"totalOxygenFlux", totalOxygenFlux},   //
        psUtils::Item{"totalIonFlux", totalIonFlux},         //
        psUtils::Item{"ionEnergy", ionEnergy},               //
        psUtils::Item{"A_O", A_O}                            //
    );
  }
};
