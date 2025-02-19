#pragma once

#include <lsAdvect.hpp>
#include <rayTrace.hpp>
#include <vcLogger.hpp>

#include <vector>

namespace viennaps {

using namespace viennacore;

using IntegrationScheme = viennals::IntegrationSchemeEnum;

template <typename NumericType, int D> struct RayTracingParameters {
  typename viennaray::TraceDirection sourceDirection =
      D == 3 ? viennaray::TraceDirection::POS_Z
             : viennaray::TraceDirection::POS_Y;
  typename viennaray::NormalizationType normalizationType =
      viennaray::NormalizationType::SOURCE;
  unsigned raysPerPoint = 1000;
  NumericType diskRadius = 0.;
  bool useRandomSeeds = true;
  bool ignoreFluxBoundaries = false;
  int smoothingNeighbors = 1;
};

template <typename NumericType> struct AdvectionParameters {
  IntegrationScheme integrationScheme =
      IntegrationScheme::ENGQUIST_OSHER_1ST_ORDER;
  NumericType timeStepRatio = 0.4999;
  NumericType dissipationAlpha = 1.0;
  bool checkDissipation = true;
  bool velocityOutput = false;
  bool ignoreVoids = false;
};

template <typename NumericType> class ProcessParams {
private:
  std::vector<NumericType> scalarData;
  std::vector<std::string> scalarDataLabels;

public:
  void insertNextScalar(NumericType value, std::string label = "scalarData") {
    scalarData.push_back(value);
    scalarDataLabels.push_back(label);
  }

  NumericType &getScalarData(int i) { return scalarData[i]; }

  const NumericType &getScalarData(int i) const { return scalarData[i]; }

  NumericType &getScalarData(std::string label) {
    int idx = getScalarDataIndex(label);
    return scalarData[idx];
  }

  int getScalarDataIndex(std::string label) {
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
  std::string getScalarDataLabel(int i) const {
    if (i >= scalarDataLabels.size())
      Logger::getInstance()
          .addError("Getting scalar data label in ProcessParams out of range.")
          .print();
    return scalarDataLabels[i];
  }
};

} // namespace viennaps
