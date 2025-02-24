#pragma once

#include <sstream>

#include <vcSmartPointer.hpp>

#include "applicationParameters.hpp"

using namespace viennaps;

class ApplicationParser {
private:
  SmartPointer<ApplicationParameters> params;

public:
  ApplicationParser() {}

  ApplicationParser(const SmartPointer<ApplicationParameters> passedParameters)
      : params(passedParameters) {}

  void
  setParameters(const SmartPointer<ApplicationParameters> passedParameters) {
    params = passedParameters;
  }

  CommandType parseCommand(std::istringstream &stream, const int lineNumber) {
    std::string command;
    stream >> command;
    if (command == "INIT") {
      std::cout << "Initializing ...\n";
      parseInit(stream);
      return CommandType::INIT;
    } else if (command == "GEOMETRY") {
      std::cout << "Creating geometry ..." << std::endl;
      parseGeometry(stream);
      return CommandType::GEOMETRY;
    } else if (command == "PROCESS") {
      std::cout << "Starting process ..." << std::endl;
      parseProcess(stream);
      return CommandType::PROCESS;
    } else if (command == "OUTPUT") {
      std::cout << "Writing geometry to file ..." << std::endl;
      parseOutput(stream);
      return CommandType::OUTPUT;
    } else if (command == "PLANARIZE") {
      std::cout << "Planarizing ..." << std::endl;
      parsePlanarize(stream);
      return CommandType::PLANARIZE;
    } else {
      std::cout << "Unknown command in config file. Skipping line "
                << lineNumber << std::endl;
      return CommandType::NONE;
    }
  }

private:
  void parseMaterial(const std::string materialString, Material &material) {
    if (materialString == "Undefined") {
      material = Material::Undefined;
    } else if (materialString == "Si") {
      material = Material::Si;
    } else if (materialString == "SiO2") {
      material = Material::SiO2;
    } else if (materialString == "Si3N4") {
      material = Material::Si3N4;
    } else if (materialString == "PolySi") {
      material = Material::PolySi;
    } else if (materialString == "Polymer") {
      material = Material::Polymer;
    } else if (materialString == "Al2O3") {
      material = Material::Al2O3;
    } else if (materialString == "SiC") {
      material = Material::SiC;
    } else if (materialString == "Metal") {
      material = Material::Metal;
    } else if (materialString == "W") {
      material = Material::W;
    } else if (materialString == "Dielectric") {
      material = Material::Dielectric;
    } else if (materialString == "SiON") {
      material = Material::SiON;
    } else if (materialString == "SiN") {
      material = Material::SiN;
    } else if (materialString == "TiN") {
      material = Material::TiN;
    } else if (materialString == "Cu") {
      material = Material::Cu;
    } else if (materialString == "Air") {
      material = Material::Air;
    } else if (materialString == "GAS") {
      material = Material::GAS;
    } else if (materialString == "GaN") {
      material = Material::GaN;
    } else {
      std::cout << "Unknown material: " << materialString << std::endl;
      material = Material::Undefined;
    }
  }

  void parseInit(std::istringstream &stream) {
    unsigned integrationSchemeNum = 0;
    auto config = parseLineStream(stream);
    utils::AssignItems(config, utils::Item{"xExtent", params->xExtent},
                       utils::Item{"yExtent", params->yExtent},
                       utils::Item{"resolution", params->gridDelta},
                       utils::Item{"logLevel", params->logLevel},
                       utils::Item{"periodic", params->periodicBoundary},
                       utils::Item{"integrationScheme", integrationSchemeNum});
    if (integrationSchemeNum > 9) {
      std::cout << "Invalid integration scheme number. Using default."
                << std::endl;
      integrationSchemeNum = 0;
    }
    params->integrationScheme =
        static_cast<viennals::IntegrationSchemeEnum>(integrationSchemeNum);
    Logger::setLogLevel(static_cast<LogLevel>(params->logLevel));
  }

  void parseGeometry(std::istringstream &stream) {
    std::string type;
    stream >> type;
    auto config = parseLineStream(stream);
    std::string material;
    if (type == "Trench") {
      params->geometryType = GeometryType::TRENCH;
      utils::AssignItems(config, utils::Item{"width", params->trenchWidth},
                         utils::Item{"depth", params->trenchHeight},
                         utils::Item{"zPos", params->maskZPos},
                         utils::Item{"tapering", params->taperAngle},
                         utils::Item{"mask", params->mask},
                         utils::Item{"material", material});
      parseMaterial(material, params->material);
    } else if (type == "Hole") {
      params->geometryType = GeometryType::HOLE;
      utils::AssignItems(config, utils::Item{"radius", params->holeRadius},
                         utils::Item{"depth", params->holeDepth},
                         utils::Item{"zPos", params->maskZPos},
                         utils::Item{"tapering", params->taperAngle},
                         utils::Item{"mask", params->mask},
                         utils::Item{"material", material});
      parseMaterial(material, params->material);
    } else if (type == "Plane") {
      params->geometryType = GeometryType::PLANE;
      utils::AssignItems(config, utils::Item{"zPos", params->maskZPos});
      parseMaterial(material, params->material);
    } else if (type == "Stack") {
      params->geometryType = GeometryType::STACK;
      utils::AssignItems(
          config, utils::Item{"numLayers", params->numLayers},
          utils::Item{"layerHeight", params->layerHeight},
          utils::Item{"maskHeight", params->maskHeight},
          utils::Item{"substrateHeight", params->substrateHeight},
          utils::Item{"holeRadius", params->holeRadius});
    } else if (type == "GDS") {
      params->geometryType = GeometryType::GDS;
      utils::AssignItems(config, utils::Item{"file", params->fileName},
                         utils::Item{"layer", params->layers},
                         utils::Item{"zPos", params->maskZPos},
                         utils::Item{"maskHeight", params->maskHeight},
                         utils::Item{"invert", params->maskInvert},
                         utils::Item{"xPadding", params->xPadding},
                         utils::Item{"yPadding", params->yPadding},
                         utils::Item{"material", material});
      parseMaterial(material, params->material);
    } else if (type == "Import") {
      params->geometryType = GeometryType::IMPORT;
      utils::AssignItems(config, utils::Item{"file", params->fileName},
                         utils::Item{"layers", params->layers});
    } else {
      params->geometryType = GeometryType::NONE;
      std::cout << "Invalid geometry type." << std::endl;
      exit(1);
    }
  }

  void parseProcess(std::istringstream &stream) {
    std::string model;
    stream >> model;
    auto config = parseLineStream(stream);
    if (model == "SingleParticleProcess") {
      std::string material;
      params->processType = ProcessType::SINGLEPARTICLEPROCESS;
      utils::AssignItems(config, utils::Item{"rate", params->rate},
                         utils::Item{"time", params->processTime},
                         utils::Item{"sticking", params->sticking},
                         utils::Item{"cosineExponent", params->cosinePower},
                         utils::Item{"smoothFlux", params->smoothFlux},
                         utils::Item{"raysPerPoint", params->raysPerPoint},
                         utils::Item{"material", material});
      parseMaterial(material, params->material);
    } else if (model == "SF6O2Etching") {
      params->processType = ProcessType::SF6O2ETCHING;
      utils::AssignItems(config, utils::Item{"time", params->processTime},
                         utils::Item{"ionFlux", params->ionFlux},
                         utils::Item{"meanIonEnergy", params->ionEnergy},
                         utils::Item{"sigmaIonEnergy", params->sigmaIonEnergy},
                         utils::Item{"etchantFlux", params->etchantFlux},
                         utils::Item{"oxygenFlux", params->oxygenFlux},
                         utils::Item{"A_O", params->A_O},
                         utils::Item{"smoothFlux", params->smoothFlux},
                         utils::Item{"ionExponent", params->ionExponent},
                         utils::Item{"raysPerPoint", params->raysPerPoint});
    } else if (model == "FluorocarbonEtching") {
      params->processType = ProcessType::FLUOROCARBONETCHING;
      utils::AssignItems(config, utils::Item{"time", params->processTime},
                         utils::Item{"ionFlux", params->ionFlux},
                         utils::Item{"meanIonEnergy", params->ionEnergy},
                         utils::Item{"sigmaIonEnergy", params->sigmaIonEnergy},
                         utils::Item{"etchantFlux", params->etchantFlux},
                         utils::Item{"polyFlux", params->oxygenFlux},
                         utils::Item{"deltaP", params->deltaP},
                         utils::Item{"smoothFlux", params->smoothFlux},
                         utils::Item{"raysPerPoint", params->raysPerPoint});
    } else if (model == "SphereDistribution") {
      params->processType = ProcessType::SPHEREDISTRIBUTION;
      utils::AssignItems(config, utils::Item{"radius", params->radius});
    } else if (model == "BoxDistribution") {
      params->processType = ProcessType::BOXDISTRIBUTION;
      utils::AssignItems(config, utils::Item{"halfAxisX", params->halfAxes[0]},
                         utils::Item{"halfAxisY", params->halfAxes[1]},
                         utils::Item{"halfAxisZ", params->halfAxes[2]});
    } else if (model == "DirectionalEtching") {
      params->processType = ProcessType::DIRECTIONALETCHING;
      std::string material = "Undefined";
      utils::AssignItems(
          config, utils::Item{"direction", params->direction},
          utils::Item{"directionalRate", params->directionalRate},
          utils::Item{"isotropicRate", params->isotropicRate},
          utils::Item{"time", params->processTime},
          utils::Item{"maskMaterial", material});
      parseMaterial(material, params->maskMaterial);
    } else if (model == "Isotropic") {
      params->processType = ProcessType::ISOTROPIC;
      std::string material = "Undefined";
      std::string maskMaterial = "Mask";
      utils::AssignItems(config, utils::Item{"rate", params->rate},
                         utils::Item{"time", params->processTime},
                         utils::Item{"time", params->processTime},
                         utils::Item{"material", material},
                         utils::Item{"maskMaterial", maskMaterial});
      parseMaterial(material, params->material);
      parseMaterial(maskMaterial, params->maskMaterial);
    } else if (model == "Anisotropic") {
      params->processType = ProcessType::ANISOTROPIC;
    } else {
      params->processType = ProcessType::NONE;
      std::cout << "Invalid process model: " << model << std::endl;
    }
  }

  void parsePlanarize(std::istringstream &stream) {
    auto config = parseLineStream(stream);
    utils::AssignItems(config, utils::Item{"height", params->maskZPos});
  }

  void parseOutput(std::istringstream &stream) {
    std::string outType;
    stream >> outType;
    std::transform(outType.begin(), outType.end(), outType.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (outType == "surface") {
      params->out = OutputType::SURFACE;
    } else if (outType == "volume") {
      params->out = OutputType::VOLUME;
    } else {
      std::cout << "Unknown output type. Using default.";
    }

    stream >> params->fileName;
  }

  std::unordered_map<std::string, std::string>
  parseLineStream(std::istringstream &input) {
    // Regex to find trailing and leading whitespaces
    const auto wsRegex = std::regex("^ +| +$|( ) +");

    // Regular expression for extracting key and value separated by '=' as two
    // separate capture groups
    const auto keyValueRegex = std::regex(
        R"rgx([ \t]*([0-9a-zA-Z_\-\.+]+)[ \t]*=[ \t]*([0-9a-zA-Z_\-\.+]+).*$)rgx");

    // Reads a simple config file containing a single <key>=<value> pair per
    // line and returns the content as an unordered map
    std::unordered_map<std::string, std::string> paramMap;
    std::string expression;
    while (input >> expression) {
      // Remove trailing and leading whitespaces
      expression = std::regex_replace(expression, wsRegex, "$1");
      // Skip this expression if it is marked as a comment
      if (expression.rfind('#') == 0 || expression.empty())
        continue;

      // Extract key and value
      std::smatch smatch;
      if (std::regex_search(expression, smatch, keyValueRegex)) {
        if (smatch.size() < 3) {
          std::cerr << "Malformed expression:\n " << expression;
          continue;
        }

        paramMap.insert({smatch[1], smatch[2]});
      }
    }
    return paramMap;
  }
};
