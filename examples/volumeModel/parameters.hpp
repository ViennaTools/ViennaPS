#pragma once

#include <string>
#include <unordered_map>

#include <psUtils.hpp>

template <typename T> struct Parameters {
  // Domain
  T gridDelta = 0.25;
  T xExtent = 15.;
  T yExtent = 7.;

  // Geometry
  T finWidth = 5.;
  T finHeight = 15.;

  // Process
  T meanFreePath = 0.75;
  T ionEnergy = 100.;

  Parameters() {}

  void fromMap(std::unordered_map<std::string, std::string> &m) {
    psUtils::AssignItems(                            //
        m,                                           //
        psUtils::Item{"gridDelta", gridDelta},       //
        psUtils::Item{"xExtent", xExtent},           //
        psUtils::Item{"yExtent", yExtent},           //
        psUtils::Item{"finWidth", finWidth},         //
        psUtils::Item{"finHeight", finHeight},       //
        psUtils::Item{"meanFreePath", meanFreePath}, //
        psUtils::Item{"ionEnergy", ionEnergy}        //
    );
  }
};
