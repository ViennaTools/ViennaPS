#pragma once

#include <string>
#include <unordered_map>

#include <psUtils.hpp>

template <typename T> struct Parameters {
  // Domain
  T gridDelta = 0.02;
  T xExtent = 1.;
  T yExtent = 1.;

  // Geometry
  T trenchWidth = 0.4;
  T trenchHeight = 0.8;
  T taperAngle = 0.;

  // Process
  T layerThickness = .15;

  Parameters() {}

  void fromMap(std::unordered_map<std::string, std::string> &m) {
    psUtils::AssignItems(                               //
        m,                                              //
        psUtils::Item{"gridDelta", gridDelta},          //
        psUtils::Item{"xExtent", xExtent},              //
        psUtils::Item{"yExtent", yExtent},              //
        psUtils::Item{"trenchWidth", trenchWidth},      //
        psUtils::Item{"trenchHeight", trenchHeight},    //
        psUtils::Item{"taperAngle", taperAngle},        //
        psUtils::Item{"layerThickness", layerThickness} //
    );
  }
};
