#pragma once

#include "../psUtil.hpp"

#include <lsAdvect.hpp>
#include <rayUtil.hpp>
#include <vcLogger.hpp>

#include <vector>

namespace viennaps {

using namespace viennacore;

using IntegrationScheme = viennals::IntegrationSchemeEnum;
using TemporalScheme = viennals::TemporalSchemeEnum;

struct RayTracingParameters {
  viennaray::NormalizationType normalizationType =
      viennaray::NormalizationType::SOURCE;
  bool ignoreFluxBoundaries = false;
  bool useRandomSeeds = true;
  unsigned rngSeed = 0;
  unsigned raysPerPoint = 1000;
  int smoothingNeighbors = 1;
  double diskRadius = 0.;
  double minNodeDistanceFactor =
      0.05; // factor of grid delta to determine min. node distance for triangle
            // mesh generation
  unsigned maxBoundaryHits = 1000;
  unsigned maxReflections = std::numeric_limits<unsigned>::max();

  auto toMetaData() const {
    std::unordered_map<std::string, std::vector<double>> metaData;
    metaData["RaysPerPoint"] = {static_cast<double>(raysPerPoint)};
    metaData["SmoothingNeighbors"] = {static_cast<double>(smoothingNeighbors)};
    metaData["DiskRadius"] = {diskRadius};
    metaData["MinNodeDistanceFactor"] = {minNodeDistanceFactor};
    metaData["MaxBoundaryHits"] = {static_cast<double>(maxBoundaryHits)};
    metaData["MaxReflections"] = {static_cast<double>(maxReflections)};
    metaData["RngSeed"] = {static_cast<double>(rngSeed)};
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
  TemporalScheme temporalScheme =
      TemporalScheme::FORWARD_EULER;
  double timeStepRatio = 0.4999;
  double dissipationAlpha = 1.0;
  double adaptiveTimeStepThreshold = 0.05;
  bool checkDissipation = true;
  bool velocityOutput = false;
  bool ignoreVoids = false;
  bool adaptiveTimeStepping = false;

  auto toMetaData() const {
    std::unordered_map<std::string, std::vector<double>> metaData;
    metaData["IntegrationScheme"] = {static_cast<double>(integrationScheme)};
    metaData["TemporalScheme"] = {static_cast<double>(temporalScheme)};
    metaData["TimeStepRatio"] = {timeStepRatio};
    metaData["DissipationAlpha"] = {dissipationAlpha};
    return metaData;
  }

  auto toMetaDataString() const {
    return "\nIntegrationScheme: " +
           util::convertIntegrationSchemeToString(integrationScheme) +
           "\nTemporalScheme: " +
           util::convertTemporalSchemeToString(temporalScheme) +
           "\nTimeStepRatio: " + std::to_string(timeStepRatio) +
           "\nDissipationAlpha: " + std::to_string(dissipationAlpha) +
           "\nCheckDissipation: " + util::boolString(checkDissipation) +
           "\nVelocityOutput: " + util::boolString(velocityOutput) +
           "\nIgnoreVoids: " + util::boolString(ignoreVoids);
  }
};

struct CoverageParameters {
  double tolerance = 0.0;
  unsigned maxIterations = std::numeric_limits<unsigned>::max();
  bool initialized = false;

  auto toMetaData() const {
    std::unordered_map<std::string, std::vector<double>> metaData;
    metaData["CoverageTolerance"] = {tolerance};
    metaData["CoverageMaxIterations"] = {static_cast<double>(maxIterations)};
    return metaData;
  }

  auto toMetaDataString() const {
    return "\nCoverageTolerance: " + std::to_string(tolerance) +
           "\nCoverageMaxIterations: " + std::to_string(maxIterations);
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
    VIENNACORE_LOG_ERROR("Can not find scalar data label in ProcessParams.");
    return -1;
  }

  std::vector<NumericType> &getScalarData() { return scalarData; }

  const std::vector<NumericType> &getScalarData() const { return scalarData; }
  [[nodiscard]] std::string getScalarDataLabel(int i) const {
    if (i >= scalarDataLabels.size()) {
      VIENNACORE_LOG_ERROR(
          "Getting scalar data label in ProcessParams out of range.");
      return "";
    }
    return scalarDataLabels[i];
  }
};

} // namespace viennaps
