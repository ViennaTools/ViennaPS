#pragma once

#include <string>
#include <unordered_map>

#include <psUtils.hpp>

template <typename T> struct Parameters {
  // Domain
  T gridDelta = 0.92;
  T xExtent = 40.0;
  T yExtent = 80.0;

  // Geometry
  T finWidth = 10.0;
  T finHeight = 10.0;
  T finLength = 60.0;

  // Process
  T processTime = 30.;
  T epitaxyRate = 10.;

  Parameters() {}

  void fromMap(std::unordered_map<std::string, std::string> &m) {
    psUtils::AssignItems(                          //
        m,                                         //
        psUtils::Item{"gridDelta", gridDelta},     //
        psUtils::Item{"xExtent", xExtent},         //
        psUtils::Item{"yExtent", yExtent},         //
        psUtils::Item{"finWidth", finWidth},       //
        psUtils::Item{"finHeight", finHeight},     //
        psUtils::Item{"finLength", finLength},     //
        psUtils::Item{"processTime", processTime}, //
        psUtils::Item{"epitaxyRate", epitaxyRate}  //
    );
  }
};
