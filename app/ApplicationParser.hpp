#pragma once

#include <sstream>

#include <psSmartPointer.hpp>

#include <ApplicationParameters.hpp>

class ApplicationParser {
private:
  psSmartPointer<ApplicationParameters> params;

public:
  ApplicationParser() {}

  ApplicationParser(
      const psSmartPointer<ApplicationParameters> passedParameters)
      : params(passedParameters) {}

  void
  setParameters(const psSmartPointer<ApplicationParameters> passedParameters) {
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
  void parseInit(std::istringstream &stream) {
    unsigned integrationSchemeNum = 0;
    auto config = parseLineStream(stream);
    psUtils::AssignItems(
        config, psUtils::Item{"xExtent", params->xExtent},
        psUtils::Item{"yExtent", params->yExtent},
        psUtils::Item{"resolution", params->gridDelta},
        psUtils::Item{"printIntermediate", params->printIntermediate},
        psUtils::Item{"periodic", params->periodicBoundary},
        psUtils::Item{"integrationScheme", integrationSchemeNum});
    if (integrationSchemeNum > 9) {
      std::cout << "Invalid integration scheme number. Using default."
                << std::endl;
      integrationSchemeNum = 0;
    }
    params->integrationScheme =
        static_cast<lsIntegrationSchemeEnum>(integrationSchemeNum);
  }

  void parseGeometry(std::istringstream &stream) {
    std::string type;
    stream >> type;
    auto config = parseLineStream(stream);
    if (type == "Trench") {
      params->geometryType = GeometryType::TRENCH;
      psUtils::AssignItems(config, psUtils::Item{"width", params->trenchWidth},
                           psUtils::Item{"depth", params->trenchHeight},
                           psUtils::Item{"zPos", params->maskZPos},
                           psUtils::Item{"tapering", params->taperAngle},
                           psUtils::Item{"mask", params->mask});
    } else if (type == "Hole") {
      params->geometryType = GeometryType::HOLE;
      psUtils::AssignItems(config, psUtils::Item{"radius", params->holeRadius},
                           psUtils::Item{"depth", params->holeDepth},
                           psUtils::Item{"zPos", params->maskZPos},
                           psUtils::Item{"tapering", params->taperAngle},
                           psUtils::Item{"mask", params->mask});
    } else if (type == "Plane") {
      params->geometryType = GeometryType::PLANE;
      psUtils::AssignItems(config, psUtils::Item{"zPos", params->maskZPos});
    } else if (type == "GDS") {
      params->geometryType = GeometryType::GDS;
      psUtils::AssignItems(config, psUtils::Item{"file", params->fileName},
                           psUtils::Item{"layer", params->layers},
                           psUtils::Item{"zPos", params->maskZPos},
                           psUtils::Item{"maskHeight", params->maskHeight},
                           psUtils::Item{"pointOrder", params->pointOrder},
                           psUtils::Item{"invert", params->maskInvert},
                           psUtils::Item{"xPadding", params->xPadding},
                           psUtils::Item{"yPadding", params->yPadding});
    } else if (type == "Import") {
      params->geometryType = GeometryType::IMPORT;
      psUtils::AssignItems(config, psUtils::Item{"file", params->fileName},
                           psUtils::Item{"layers", params->layers});
    } else {
      params->geometryType = GeometryType::NONE;
      std::cout << "Invalid mask type." << std::endl;
      exit(1);
    }
  }

  void parseProcess(std::istringstream &stream) {
    std::string model;
    stream >> model;
    auto config = parseLineStream(stream);
    if (model == "Deposition") {
      params->processType = ProcessType::DEPOSITION;
      psUtils::AssignItems(config, psUtils::Item{"rate", params->rate},
                           psUtils::Item{"time", params->processTime},
                           psUtils::Item{"sticking", params->sticking},
                           psUtils::Item{"cosineExponent", params->cosinePower},
                           psUtils::Item{"raysPerPoint", params->raysPerPoint});
    } else if (model == "SF6O2Etching") {
      params->processType = ProcessType::SF6O2ETCHING;
      psUtils::AssignItems(
          config, psUtils::Item{"time", params->processTime},
          psUtils::Item{"ionFlux", params->totalIonFlux},
          psUtils::Item{"ionEnergy", params->ionEnergy},
          psUtils::Item{"etchantFlux", params->totalEtchantFlux},
          psUtils::Item{"oxygenFlux", params->totalOxygenFlux},
          psUtils::Item{"A_O", params->A_O},
          psUtils::Item{"raysPerPoint", params->raysPerPoint},
          psUtils::Item{"maskId", params->maskId});
    } else if (model == "SphereDistribution") {
      params->processType = ProcessType::SPHEREDISTRIBUTION;
      psUtils::AssignItems(config, psUtils::Item{"radius", params->radius});
    } else if (model == "BoxDistribution") {
      params->processType = ProcessType::BOXDISTRIBUTION;
      psUtils::AssignItems(config,
                           psUtils::Item{"halfAxisX", params->halfAxes[0]},
                           psUtils::Item{"halfAxisY", params->halfAxes[1]},
                           psUtils::Item{"halfAxisZ", params->halfAxes[2]});
    } else if (model == "DirectionalEtching") {
      params->processType = ProcessType::DIRECTIONALETCHING;
      psUtils::AssignItems(
          config, psUtils::Item{"direction", params->direction},
          psUtils::Item{"directionalRate", params->directionalRate},
          psUtils::Item{"isotropicRate", params->isotropicRate},
          psUtils::Item{"time", params->processTime},
          psUtils::Item{"maskId", params->maskId});
    } else if (model == "Isotropic") {
      params->processType = ProcessType::ISOTROPIC;
      psUtils::AssignItems(config, psUtils::Item{"rate", params->rate},
                           psUtils::Item{"time", params->processTime},
                           psUtils::Item{"maskId", params->maskId});
    } else if (model == "WetEtching") {
      params->processType = ProcessType::WETETCHING;
      psUtils::AssignItems(config, psUtils::Item{"time", params->processTime},
                           psUtils::Item{"maskId", params->maskId});
    } else {
      params->processType = ProcessType::NONE;
      std::cout << "Invalid process model: " << model << std::endl;
    }
  }

  void parsePlanarize(std::istringstream &stream) {
    auto config = parseLineStream(stream);
    psUtils::AssignItems(config, psUtils::Item{"height", params->maskZPos});
  }

  void parseOutput(std::istringstream &stream) { stream >> params->fileName; }

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
