#pragma once

#include <string>
#include <unordered_map>

#include <psUtil.hpp>

struct Parameters {
  // Domain
  double gridDelta = 2.5;
  int periodicBoundary = 0;

  // Geometry
  int numLayers = 12;
  int numPillars = 5;
  double layerHeight = 20.0;
  double lateralSpacing = 200.;

  double maskHeight = 50.;
  double maskWidth = 100.;

  double trenchWidth = 100.;
  double trenchWidthBot = 60.;

  double overEtch = 100.;
  double offSet = 0.;

  // Process
  // double rate_SiGe = 4.;
  // double rate_Si = 0.;
  // double rate_SiO2 = 0.;

  // double time = 10.;
  // double sticking = 0.05;
  // double power = 1.;
  // int numRaysPerPoint = 500;

  // Utils
  int saveVolume = 1;
  int halfGeometry = 1;

  // double measureHeight = 10.;
  // double measureHeightDelta = 10.;
  double pillarMaxPos = -300.;

  std::string fileName = "default_";
  std::string pathFile = "";

  Parameters() {}

  std::array<double, 2> getExtent() const {
    std::array<double, 2> extent;

    extent[0] = 2 * lateralSpacing + numPillars * maskWidth +
                (numPillars - 1) * trenchWidth;
    extent[1] = overEtch + numLayers * layerHeight + maskHeight;

    return extent;
  }

  void fromMap(std::unordered_map<std::string, std::string> &m) {
    viennaps::util::AssignItems(                                        //
        m,                                                              //
        viennaps::util::Item{"gridDelta", gridDelta},                   //
        viennaps::util::Item{"periodicBoundary", periodicBoundary},     //
        viennaps::util::Item{"lateralSpacing", lateralSpacing},         //
        viennaps::util::Item{"numLayers", numLayers},                   //
        viennaps::util::Item{"numPillars", numPillars},                 //
        viennaps::util::Item{"layerHeight", layerHeight},               //
        viennaps::util::Item{"maskHeight", maskHeight},                 //
        viennaps::util::Item{"maskWidth", maskWidth},                   //
        viennaps::util::Item{"trenchWidth", trenchWidth},               //
        viennaps::util::Item{"overEtch", overEtch},                     //
        viennaps::util::Item{"offSet", offSet},                         //
        viennaps::util::Item{"trenchWidthBot", trenchWidthBot},         //
        // viennaps::util::Item{"rate_SiGe", rate_SiGe},                   //
        // viennaps::util::Item{"rate_SiO2", rate_SiO2},                   //
        // viennaps::util::Item{"rate_Si", rate_Si},                       //
        // viennaps::util::Item{"sticking", sticking},                     //
        // viennaps::util::Item{"power", power},                           //
        // viennaps::util::Item{"time", time},                             //
        viennaps::util::Item{"fileName", fileName},                     //
        viennaps::util::Item{"pathFile", pathFile},                     //
        // viennaps::util::Item{"numRaysPerPoint", numRaysPerPoint},       //
        viennaps::util::Item{"saveVolume", saveVolume},                 //
        viennaps::util::Item{"halfGeometry", halfGeometry},             //
        // viennaps::util::Item{"measureHeight", measureHeight},           //
        // viennaps::util::Item{"measureHeightDelta", measureHeightDelta}, //
        viennaps::util::Item{"pillarMaxPos", pillarMaxPos}              //
    );
  }
};