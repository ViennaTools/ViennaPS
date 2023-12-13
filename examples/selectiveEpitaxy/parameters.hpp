#pragma once

#include <string>
#include <unordered_map>

#include <psUtils.hpp>

template <typename T> struct Parameters {
  // Domain
  T gridDelta = 0.04;
  T xExtent = 2.;
  T yExtent = 2.5;

  // Geometry
  T finWidth = 0.4;
  T finHeight = 1.2;
  T finLength = 1.;
  T maskHeight = 0.4;

  // Process
  T processTime = 30.;

  Parameters() {}

  void fromMap(std::unordered_map<std::string, std::string> &m) {
    psUtils::AssignItems(                         //
        m,                                        //
        psUtils::Item{"gridDelta", gridDelta},    //
        psUtils::Item{"xExtent", xExtent},        //
        psUtils::Item{"yExtent", yExtent},        //
        psUtils::Item{"finWidth", finWidth},      //
        psUtils::Item{"finHeight", finHeight},    //
        psUtils::Item{"finLength", finLength},    //
        psUtils::Item{"maskHeight", maskHeight},  //
        psUtils::Item{"processTime", processTime} //
    );
  }
};
