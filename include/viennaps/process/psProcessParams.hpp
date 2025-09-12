#pragma once

#include "../psUtil.hpp"

#include <lsAdvect.hpp>
#include <rayTrace.hpp>
#include <vcLogger.hpp>

#include <vector>

namespace viennaps {

using namespace viennacore;

using IntegrationScheme = viennals::IntegrationSchemeEnum;

template <int D> struct RayTracingParameters {
  viennaray::TraceDirection sourceDirection =
      D == 3 ? viennaray::TraceDirection::POS_Z
             : viennaray::TraceDirection::POS_Y;
  viennaray::NormalizationType normalizationType =
      viennaray::NormalizationType::SOURCE;
  bool ignoreFluxBoundaries = false;
  bool useRandomSeeds = true;
  unsigned raysPerPoint = 1000;
  int smoothingNeighbors = 1;
  double diskRadius = 0.;
  double minNodeDistanceFactor =
      0.05; // factor of grid delta to determine min. node distance for triange
            // mesh generation

  auto toMetaData() const {
    std::unordered_map<std::string, std::vector<double>> metaData;
    metaData["RaysPerPoint"] = {static_cast<double>(raysPerPoint)};
    metaData["SmoothingNeighbors"] = {static_cast<double>(smoothingNeighbors)};
    metaData["DiskRadius"] = {diskRadius};
    metaData["MinNodeDistanceFactor"] = {minNodeDistanceFactor};
    return metaData;
  }

  auto toMetaDataString() const {
    return "\nRaysPerPoint: " + std::to_string(raysPerPoint) +
           "\nDiskRadius: " + std::to_string(diskRadius) +
           "\nSmoothingNeighbors: " + std::to_string(smoothingNeighbors) +
           "\nMinNodeDistanceFactor: " + std::to_string(minNodeDistanceFactor) +
           "\nUseRandomSeeds: " + util::boolString(useRandomSeeds);
  }
};

struct AdvectionParameters {
  IntegrationScheme integrationScheme =
      IntegrationScheme::ENGQUIST_OSHER_1ST_ORDER;
  double timeStepRatio = 0.4999;
  double dissipationAlpha = 1.0;
  bool checkDissipation = true;
  bool velocityOutput = false;
  bool ignoreVoids = false;

  auto toMetaData() const {
    std::unordered_map<std::string, std::vector<double>> metaData;
    metaData["IntegrationScheme"] = {static_cast<double>(integrationScheme)};
    metaData["TimeStepRatio"] = {timeStepRatio};
    metaData["DissipationAlpha"] = {dissipationAlpha};
    return metaData;
  }

  auto toMetaDataString() const {
    return "\nIntegrationScheme: " +
           util::convertIntegrationSchemeToString(integrationScheme) +
           "\nTimeStepRatio: " + std::to_string(timeStepRatio) +
           "\nDissipationAlpha: " + std::to_string(dissipationAlpha) +
           "\nCheckDissipation: " + util::boolString(checkDissipation) +
           "\nVelocityOutput: " + util::boolString(velocityOutput) +
           "\nIgnoreVoids: " + util::boolString(ignoreVoids);
  }
};

struct CoverageParameters {
  double coverageDeltaThreshold = 0.0;
  unsigned maxIterations = std::numeric_limits<unsigned>::max();
  bool initialized = false;

  auto toMetaData() const {
    std::unordered_map<std::string, std::vector<double>> metaData;
    metaData["CoverageDeltaThreshold"] = {coverageDeltaThreshold};
    metaData["MaxIterations"] = {static_cast<double>(maxIterations)};
    return metaData;
  }

  auto toMetaDataString() const {
    return "\nCoverageDeltaThreshold: " +
           std::to_string(coverageDeltaThreshold) +
           "\nMaxIterations: " + std::to_string(maxIterations);
  }
};

struct AtomicLayerProcessParameters {
  unsigned numCycles = 1;
  double pulseTime = 1.0;
  double coverageTimeStep = 1.0;
  double purgePulseTime = 0.0;

  auto toMetaData() const {
    std::unordered_map<std::string, std::vector<double>> metaData;
    metaData["NumCycles"] = {static_cast<double>(numCycles)};
    metaData["PulseTime"] = {pulseTime};
    metaData["CoverageTimeStep"] = {coverageTimeStep};
    metaData["PurgePulseTime"] = {purgePulseTime};
    return metaData;
  }

  auto toMetaDataString() const {
    return "\nNumCycles: " + std::to_string(numCycles) +
           "\nPulseTime: " + std::to_string(pulseTime) +
           "\nCoverageTimeStep: " + std::to_string(coverageTimeStep) +
           "\nPurgePulseTime: " + std::to_string(purgePulseTime);
  }
};

template <typename NumericType> class ProcessParams {
private:
  std::vector<NumericType> scalarData;
  std::vector<std::string> scalarDataLabels;

public:
  void insertNextScalar(NumericType value,
                        const std::string &label = "scalarData") {
    scalarData.push_back(value);
    scalarDataLabels.push_back(label);
  }

  NumericType &getScalarData(int i) { return scalarData[i]; }

  const NumericType &getScalarData(int i) const { return scalarData[i]; }

  NumericType &getScalarData(const std::string &label) {
    int idx = getScalarDataIndex(label);
    return scalarData[idx];
  }

  [[nodiscard]] int getScalarDataIndex(const std::string &label) const {
    for (int i = 0; i < scalarDataLabels.size(); ++i) {
      if (scalarDataLabels[i] == label) {
        return i;
      }
    }
    Logger::getInstance()
        .addError("Can not find scalar data label in ProcessParams.")
        .print();
    return -1;
  }

  std::vector<NumericType> &getScalarData() { return scalarData; }

  const std::vector<NumericType> &getScalarData() const { return scalarData; }
  [[nodiscard]] std::string getScalarDataLabel(int i) const {
    if (i >= scalarDataLabels.size())
      Logger::getInstance()
          .addError("Getting scalar data label in ProcessParams out of range.")
          .print();
    return scalarDataLabels[i];
  }
};

} // namespace viennaps
