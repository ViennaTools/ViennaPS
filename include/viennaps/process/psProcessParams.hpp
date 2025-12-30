#pragma once

#include "../psUtil.hpp"

#include <lsAdvect.hpp>
#include <rayUtil.hpp>
#include <vcLogger.hpp>

#include <vector>

namespace viennaps {

using namespace viennacore;

using SpatialScheme = viennals::SpatialSchemeEnum;
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
    auto metaData = util::metaDataToString(toMetaData());
    return metaData + "\nUseRandomSeeds: " + util::toString(useRandomSeeds) +
           "\nIgnoreFluxBoundaries: " + util::toString(ignoreFluxBoundaries);
  }
};

struct AdvectionParameters {
  // Union to support legacy integrationScheme parameter
  // will be removed in a future release
  union {
    SpatialScheme spatialScheme = SpatialScheme::ENGQUIST_OSHER_1ST_ORDER;
    [[deprecated("Use spatialScheme instead")]] SpatialScheme integrationScheme;
  };
  // Legacy type alias for the integration scheme
  // will be removed in a future release
  using IntegrationScheme [[deprecated("Use SpatialScheme instead")]] =
      SpatialScheme;

  TemporalScheme temporalScheme = TemporalScheme::FORWARD_EULER;
  double timeStepRatio = 0.4999;
  double dissipationAlpha = 1.0;
  unsigned adaptiveTimeStepSubdivisions = 20;
  bool checkDissipation = true;
  bool velocityOutput = false;
  bool ignoreVoids = false;
  bool adaptiveTimeStepping = false;
  bool calculateIntermediateVelocities = false;

  auto toMetaData() const {
    std::unordered_map<std::string, std::vector<double>> metaData;
    metaData["SpatialScheme"] = {static_cast<double>(spatialScheme)};
    metaData["TemporalScheme"] = {static_cast<double>(temporalScheme)};
    metaData["TimeStepRatio"] = {timeStepRatio};
    metaData["DissipationAlpha"] = {dissipationAlpha};
    return metaData;
  }

  auto toMetaDataString() const {
    return "\nSpatialScheme: " +
           util::convertSpatialSchemeToString(spatialScheme) +
           "\nTemporalScheme: " + util::toString(temporalScheme) +
           "\nTimeStepRatio: " + util::toString(timeStepRatio) +
           "\nDissipationAlpha: " + util::toString(dissipationAlpha) +
           "\nCheckDissipation: " + util::toString(checkDissipation) +
           "\nVelocityOutput: " + util::toString(velocityOutput) +
           "\nIgnoreVoids: " + util::toString(ignoreVoids) +
           "\nAdaptiveTimeStepping: " + util::toString(adaptiveTimeStepping) +
           "\nCalculateIntermediateVelocities: " + util::toString(calculateIntermediateVelocities);
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

  auto toMetaDataString() const { return util::metaDataToString(toMetaData()); }
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

  auto toMetaDataString() const { return util::metaDataToString(toMetaData()); }
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
