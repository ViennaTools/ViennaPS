#pragma once

#include <fstream>
#include <iostream>
#include <istream>
#include <regex>
#include <string>
#include <unordered_map>

template <typename NumericType> struct psProcessParameters {
  // Geometry parameters
  NumericType xExtent = 1.;
  NumericType yExtent = 1.;
  NumericType gridDelta = 0.02;

  NumericType maskHeight = 0.2;
  NumericType taperAngle = 0.;

  NumericType holeRadius = 0.2;
  NumericType trenchWidth = 0.2;
  NumericType trenchHeight = 0.4;
  NumericType finWidth = 0.5;
  NumericType finHeight = 0.7;

  // Common process parameters
  NumericType processTime = 200;
  NumericType ionEnergy = 100.;

  // Simple deposition parameters
  NumericType stickingProbability = 0.1;
  NumericType sourcePower = 1.;

  // SF6O2 parameters
  int P, y;
  NumericType totalEtchantFlux = 4.5e16;
  NumericType totalOxygenFlux = 1.e18;
  NumericType totalIonFlux = 2e16;
  NumericType A_O = 3.;

  // Plasma damage parameters
  NumericType meanFreePath = .1;

  void print() const {
    std::cout << "Process paramters:"
              << "\n\t xExtent: " << xExtent << "\n\t yExtent: " << yExtent
              << "\n\t gridDelta: " << gridDelta
              << "\n\t maskHeight: " << maskHeight
              << "\n\t taperAngle: " << taperAngle
              << "\n\t holeRadius: " << holeRadius
              << "\n\t trenchWidth: " << trenchWidth
              << "\n\t trenchHeight: " << trenchHeight
              << "\n\t processTime: " << processTime
              << "\n\t ionEnergy: " << ionEnergy
              << "\n\t stickingProbability: " << stickingProbability
              << "\n\t sourcePower: " << sourcePower
              << "\n\t totalEtchantFlux: " << totalEtchantFlux
              << "\n\t totalOxygenFlux: " << totalOxygenFlux
              << "\n\t totalIonFlux: " << totalIonFlux << "\n\t A_O: " << A_O
              << "\n\t meanFreePath: " << meanFreePath << std::endl;
  }
};

template <typename NumericType> class psConfigParser {
public:
  psConfigParser() {}
  psConfigParser(std::string passedfileName) : fileName(passedfileName) {}

  void apply() {
    auto config = parseConfig(fileName);
    params = getParameters(config);
  }

  psProcessParameters<NumericType> getParameters() const { return params; }

private:
  std::string fileName;
  psProcessParameters<NumericType> params;

  std::unordered_map<std::string, NumericType>
  parseConfig(std::string filename) {
    std::ifstream f(filename);
    if (!f.is_open()) {
      std::cout << "Failed to open config file" << std::endl;
      return {};
    }
    const std::string regexPattern =
        "^([a-zA-Z_]+)[\\ \\t]*=[\\ \\t]*([0-9e\\.\\-\\+]+)";
    const std::regex rgx(regexPattern);

    std::unordered_map<std::string, NumericType> paramMap;
    std::string line;
    while (std::getline(f, line)) {
      if (line[0] == '#')
        continue;
      std::smatch smatch;
      if (std::regex_search(line, smatch, rgx)) {
        if (smatch.size() < 3) {
          std::cerr << "Malformed line:\n " << line;
          continue;
        }

        NumericType value;
        try {
          value = std::stof(smatch[2]);
        } catch (std::exception e) {
          std::cerr << "Error parsing value in line:\n " << line;
          continue;
        }
        paramMap.insert({smatch[1], value});
      }
    }
    return paramMap;
  }

  psProcessParameters<NumericType>
  getParameters(std::unordered_map<std::string, NumericType> &config) {
    psProcessParameters<NumericType> params;

    if (config.size() == 0) {
      std::cerr << "Empty config provided" << std::endl;
      return params;
    }
    for (auto [key, value] : config) {

      if (key == "xExtent") {
        params.xExtent = value;
      } else if (key == "yExtent") {
        params.yExtent = value;
      } else if (key == "gridDelta") {
        params.gridDelta = value;
      } else if (key == "maskHeight") {
        params.maskHeight = value;
      } else if (key == "taperAngle") {
        params.taperAngle = value;
      } else if (key == "holeRadius") {
        params.holeRadius = value;
      } else if (key == "trenchWidth") {
        params.trenchWidth = value;
      } else if (key == "trenchHeight") {
        params.trenchHeight = value;
      } else if (key == "finWidth") {
        params.finWidth = value;
      } else if (key == "finHeight") {
        params.finHeight = value;
      } else if (key == "processTime") {
        params.processTime = value;
      } else if (key == "ionEnergy") {
        params.ionEnergy = value;
      } else if (key == "stickingProbability") {
        params.stickingProbability = value;
      } else if (key == "sourcePower") {
        params.sourcePower = value;
      } else if (key == "P") {
        params.P = value;
      } else if (key == "y") {
        params.y = value;
      } else if (key == "totalEtchantFlux") {
        params.totalEtchantFlux = value;
      } else if (key == "totalOxygenFlux") {
        params.totalOxygenFlux = value;
      } else if (key == "totalIonFlux") {
        params.totalIonFlux = value;
      } else if (key == "A_O") {
        params.A_O = value;
      } else if (key == "meanFreePath") {
        params.meanFreePath = value;
      }
    }

    return params;
  }
};
