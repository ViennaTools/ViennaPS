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
  double buffer = 0.;

  // Utils
  int saveVolume = 1;
  int halfGeometry = 1;

  std::string fileName = "default_";
  std::string pathFile = "";
  std::string targetFile = "";

  Parameters() {}

  std::array<double, 2> getExtent() const {
    std::array<double, 2> extent;

    if (!pathFile.empty()) {
      // Extract x extent from CSV with buffer
      std::ifstream file(pathFile);
      double x, y;
      double xMin = std::numeric_limits<double>::max();
      double xMax = std::numeric_limits<double>::lowest();

      while (file >> x && file.get() == ',' && file >> y) {
        if (x < xMin)
          xMin = x;
        if (x > xMax)
          xMax = x;
      }

      // Apply buffer on both sides
      xMin -= buffer;
      xMax += buffer;
      extent = {xMax - xMin, 0.}; // Only x extent matters
    } else {
      extent[0] = 2 * lateralSpacing + numPillars * maskWidth +
                  (numPillars - 1) * trenchWidth;
      extent[1] = overEtch + numLayers * layerHeight + maskHeight;
    }
    return extent;
  }

  void fromMap(std::unordered_map<std::string, std::string> &m) {
    viennaps::util::AssignItems(                                    //
        m,                                                          //
        viennaps::util::Item{"gridDelta", gridDelta},               //
        viennaps::util::Item{"periodicBoundary", periodicBoundary}, //
        viennaps::util::Item{"lateralSpacing", lateralSpacing},     //
        viennaps::util::Item{"numLayers", numLayers},               //
        viennaps::util::Item{"numPillars", numPillars},             //
        viennaps::util::Item{"layerHeight", layerHeight},           //
        viennaps::util::Item{"maskHeight", maskHeight},             //
        viennaps::util::Item{"maskWidth", maskWidth},               //
        viennaps::util::Item{"trenchWidth", trenchWidth},           //
        viennaps::util::Item{"overEtch", overEtch},                 //
        viennaps::util::Item{"offSet", offSet},                     //
        viennaps::util::Item{"buffer", buffer},                     //
        viennaps::util::Item{"trenchWidthBot", trenchWidthBot},     //
        viennaps::util::Item{"fileName", fileName},                 //
        viennaps::util::Item{"pathFile", pathFile},                 //
        viennaps::util::Item{"targetFile", targetFile},             //
        viennaps::util::Item{"saveVolume", saveVolume},             //
        viennaps::util::Item{"halfGeometry", halfGeometry}          //
    );
  }
};