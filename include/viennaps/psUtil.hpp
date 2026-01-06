#pragma once

#include <lsAdvect.hpp>
#include <lsDomain.hpp>
#include <rayBoundary.hpp>
#include <vcLogger.hpp>

#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>

#include "psMaterials.hpp"

// Use viennacore here to avoid conflicts with other namespaces
namespace viennacore::util {
[[nodiscard]] inline viennals::SpatialSchemeEnum
convertSpatialScheme(const std::string &s) {
  if (s == "ENGQUIST_OSHER_1ST_ORDER" || s == "EO_1")
    return viennals::SpatialSchemeEnum::ENGQUIST_OSHER_1ST_ORDER;
  if (s == "ENGQUIST_OSHER_2ND_ORDER" || s == "EO_2")
    return viennals::SpatialSchemeEnum::ENGQUIST_OSHER_2ND_ORDER;
  if (s == "LAX_FRIEDRICHS_1ST_ORDER" || s == "LF_1")
    return viennals::SpatialSchemeEnum::LAX_FRIEDRICHS_1ST_ORDER;
  if (s == "LAX_FRIEDRICHS_2ND_ORDER" || s == "LF_2")
    return viennals::SpatialSchemeEnum::LAX_FRIEDRICHS_2ND_ORDER;
  if (s == "LOCAL_LAX_FRIEDRICHS_ANALYTICAL_1ST_ORDER" || s == "LLFA_1")
    return viennals::SpatialSchemeEnum::
        LOCAL_LAX_FRIEDRICHS_ANALYTICAL_1ST_ORDER;
  if (s == "LOCAL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER" || s == "LLLF_1")
    return viennals::SpatialSchemeEnum::LOCAL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER;
  if (s == "LOCAL_LOCAL_LAX_FRIEDRICHS_2ND_ORDER" || s == "LLLF_2")
    return viennals::SpatialSchemeEnum::LOCAL_LOCAL_LAX_FRIEDRICHS_2ND_ORDER;
  if (s == "LOCAL_LAX_FRIEDRICHS_1ST_ORDER" || s == "LLF_1")
    return viennals::SpatialSchemeEnum::LOCAL_LAX_FRIEDRICHS_1ST_ORDER;
  if (s == "LOCAL_LAX_FRIEDRICHS_2ND_ORDER" || s == "LLF_2")
    return viennals::SpatialSchemeEnum::LOCAL_LAX_FRIEDRICHS_2ND_ORDER;
  if (s == "STENCIL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER" || s == "SLLF_1")
    return viennals::SpatialSchemeEnum::STENCIL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER;
  if (s == "WENO_5TH_ORDER" || s == "WENO_5")
    return viennals::SpatialSchemeEnum::WENO_5TH_ORDER;
  throw std::invalid_argument(
      "The value must be one of the following: "
      "ENGQUIST_OSHER_1ST_ORDER, ENGQUIST_OSHER_2ND_ORDER, "
      "LAX_FRIEDRICHS_1ST_ORDER, LAX_FRIEDRICHS_2ND_ORDER, "
      "LOCAL_LAX_FRIEDRICHS_ANALYTICAL_1ST_ORDER, "
      "LOCAL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER, "
      "LOCAL_LOCAL_LAX_FRIEDRICHS_2ND_ORDER, "
      "LOCAL_LAX_FRIEDRICHS_1ST_ORDER, "
      "LOCAL_LAX_FRIEDRICHS_2ND_ORDER, "
      "STENCIL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER, "
      "WENO_5TH_ORDER");
}

// Helper for legacy integrationScheme parameter
// will be removed in a future release
[[nodiscard]] [[deprecated(
    "Use convertSpatialScheme instead")]] inline viennals::SpatialSchemeEnum
convertIntegrationScheme(const std::string &s) {
  VIENNACORE_LOG_WARNING("The parameter 'integrationScheme' is deprecated "
                         "and will be removed in a future release. "
                         "Please use 'spatialScheme' instead.");
  return convertSpatialScheme(s);
}

[[nodiscard]] inline viennals::TemporalSchemeEnum
convertTemporalScheme(const std::string &s) {
  if (s == "FORWARD_EULER" || s == "FE")
    return viennals::TemporalSchemeEnum::FORWARD_EULER;
  if (s == "RUNGE_KUTTA_2ND_ORDER" || s == "RK2")
    return viennals::TemporalSchemeEnum::RUNGE_KUTTA_2ND_ORDER;
  if (s == "RUNGE_KUTTA_3RD_ORDER" || s == "RK3")
    return viennals::TemporalSchemeEnum::RUNGE_KUTTA_3RD_ORDER;
  throw std::invalid_argument("The value must be one of the following: "
                              "FORWARD_EULER, RUNGE_KUTTA_2ND_ORDER, "
                              "RUNGE_KUTTA_3RD_ORDER");
}

[[nodiscard]] inline std::string
convertSpatialSchemeToString(viennals::SpatialSchemeEnum scheme) {
  switch (scheme) {
  case viennals::SpatialSchemeEnum::ENGQUIST_OSHER_1ST_ORDER:
    return "ENGQUIST_OSHER_1ST_ORDER";
  case viennals::SpatialSchemeEnum::ENGQUIST_OSHER_2ND_ORDER:
    return "ENGQUIST_OSHER_2ND_ORDER";
  case viennals::SpatialSchemeEnum::LAX_FRIEDRICHS_1ST_ORDER:
    return "LAX_FRIEDRICHS_1ST_ORDER";
  case viennals::SpatialSchemeEnum::LAX_FRIEDRICHS_2ND_ORDER:
    return "LAX_FRIEDRICHS_2ND_ORDER";
  case viennals::SpatialSchemeEnum::LOCAL_LAX_FRIEDRICHS_ANALYTICAL_1ST_ORDER:
    return "LOCAL_LAX_FRIEDRICHS_ANALYTICAL_1ST_ORDER";
  case viennals::SpatialSchemeEnum::LOCAL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER:
    return "LOCAL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER";
  case viennals::SpatialSchemeEnum::LOCAL_LOCAL_LAX_FRIEDRICHS_2ND_ORDER:
    return "LOCAL_LOCAL_LAX_FRIEDRICHS_2ND_ORDER";
  case viennals::SpatialSchemeEnum::LOCAL_LAX_FRIEDRICHS_1ST_ORDER:
    return "LOCAL_LAX_FRIEDRICHS_1ST_ORDER";
  case viennals::SpatialSchemeEnum::LOCAL_LAX_FRIEDRICHS_2ND_ORDER:
    return "LOCAL_LAX_FRIEDRICHS_2ND_ORDER";
  case viennals::SpatialSchemeEnum::STENCIL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER:
    return "STENCIL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER";
  case viennals::SpatialSchemeEnum::WENO_5TH_ORDER:
    return "WENO_5TH_ORDER";
  default:
    throw std::invalid_argument("Unknown discretization scheme.");
  }
}

[[nodiscard]] inline std::string
convertTemporalSchemeToString(viennals::TemporalSchemeEnum scheme) {
  switch (scheme) {
  case viennals::TemporalSchemeEnum::FORWARD_EULER:
    return "FORWARD_EULER";
  case viennals::TemporalSchemeEnum::RUNGE_KUTTA_2ND_ORDER:
    return "RUNGE_KUTTA_2ND_ORDER";
  case viennals::TemporalSchemeEnum::RUNGE_KUTTA_3RD_ORDER:
    return "RUNGE_KUTTA_3RD_ORDER";
  default:
    throw std::invalid_argument("Unknown temporal integration scheme.");
  }
}

[[nodiscard]] inline viennaray::BoundaryCondition convertBoundaryCondition(
    viennals::BoundaryConditionEnum originalBoundaryCondition) {
  switch (originalBoundaryCondition) {
  case viennals::BoundaryConditionEnum::REFLECTIVE_BOUNDARY:
    return viennaray::BoundaryCondition::REFLECTIVE_BOUNDARY;

  case viennals::BoundaryConditionEnum::INFINITE_BOUNDARY:
    return viennaray::BoundaryCondition::IGNORE_BOUNDARY;

  case viennals::BoundaryConditionEnum::PERIODIC_BOUNDARY:
    return viennaray::BoundaryCondition::PERIODIC_BOUNDARY;

  case viennals::BoundaryConditionEnum::POS_INFINITE_BOUNDARY:
  case viennals::BoundaryConditionEnum::NEG_INFINITE_BOUNDARY:
    return viennaray::BoundaryCondition::IGNORE_BOUNDARY;
  }
  return viennaray::BoundaryCondition::IGNORE_BOUNDARY;
}

template <typename T> [[nodiscard]] std::string toString(const T &value) {
  if constexpr (std::is_same_v<T, bool>)
    return value ? "true" : "false";
  else if constexpr (std::is_same_v<T, viennals::SpatialSchemeEnum>)
    return convertSpatialSchemeToString(
        static_cast<viennals::SpatialSchemeEnum>(value));
  else if constexpr (std::is_same_v<T, viennals::TemporalSchemeEnum>)
    return convertTemporalSchemeToString(
        static_cast<viennals::TemporalSchemeEnum>(value));
  else if constexpr (std::is_same_v<T, viennaps::Material>) {
    std::string mat = viennaps::to_string_view(value);
    return mat;
  } else if constexpr (std::is_same_v<T, std::string>)
    return value;
  else
    return std::to_string(value);
}

[[nodiscard]] inline std::string metaDataToString(
    const std::unordered_map<std::string, std::vector<double>> &metaData) {
  std::stringstream str;
  for (const auto &item : metaData) {
    str << "\n" << item.first << ": ";
    for (const auto &value : item.second) {
      str << value << " ";
    }
  }
  return str.str();
}

[[nodiscard]] inline std::array<double, 3>
hexToRGBArray(const uint32_t hexColor) {
  std::array<double, 3> rgb{};
  rgb[0] = static_cast<double>((hexColor >> 16) & 0xFF) / 255.0;
  rgb[1] = static_cast<double>((hexColor >> 8) & 0xFF) / 255.0;
  rgb[2] = static_cast<double>(hexColor & 0xFF) / 255.0;
  return rgb;
}
}; // namespace viennacore::util
