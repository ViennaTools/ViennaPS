#pragma once

#include <lsAdvect.hpp>
#include <lsDomain.hpp>
#include <rayBoundary.hpp>
#include <vcLogger.hpp>

#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>

#include "materials/psMaterialMap.hpp"
#include "psDomainSetup.hpp"

namespace viennaps {
enum class FluxEngineType {
  AUTO,         // Automatic selection
  CPU_DISK,     // CPU, Disk-based
  CPU_TRIANGLE, // CPU, Triangle-based
  GPU_DISK,     // GPU, Disk-based
  GPU_TRIANGLE, // GPU, Triangle-based
  GPU_LINE      // GPU, Line-based
};

enum class OxidantType { DRY, WET };
enum class SiliconOrientation { Si100, Si110, Si111, PolySi };
} // namespace viennaps

// Use viennacore here to avoid conflicts with other namespaces
namespace viennacore::util {

namespace detail {
std::string lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return value;
}

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
  if (s == "WENO_3RD_ORDER" || s == "WENO_3")
    return viennals::SpatialSchemeEnum::WENO_3RD_ORDER;
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
      "WENO_3RD_ORDER, "
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

[[nodiscard]] inline viennaps::FluxEngineType
convertFluxEngineType(const std::string &s) {
  if (s == "AUTO")
    return viennaps::FluxEngineType::AUTO;
  if (s == "CPU_DISK" || s == "CD")
    return viennaps::FluxEngineType::CPU_DISK;
  if (s == "CPU_TRIANGLE" || s == "CT")
    return viennaps::FluxEngineType::CPU_TRIANGLE;
  if (s == "GPU_DISK" || s == "GD")
    return viennaps::FluxEngineType::GPU_DISK;
  if (s == "GPU_TRIANGLE" || s == "GT")
    return viennaps::FluxEngineType::GPU_TRIANGLE;
  if (s == "GPU_LINE" || s == "GL")
    return viennaps::FluxEngineType::GPU_LINE;
  throw std::invalid_argument("Unknown FluxEngineType: " + s);
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

[[nodiscard]] inline viennahrle::BoundaryType
convertBoundaryType(const std::string &s) {
  const auto n = lower(s);
  if (n == "reflective_boundary" || n == "reflective")
    return viennahrle::BoundaryType::REFLECTIVE_BOUNDARY;
  if (n == "infinite_boundary" || n == "infinite")
    return viennahrle::BoundaryType::INFINITE_BOUNDARY;
  if (n == "periodic_boundary" || n == "periodic")
    return viennahrle::BoundaryType::PERIODIC_BOUNDARY;
  throw std::invalid_argument("The value must be one of the following: "
                              "REFLECTIVE_BOUNDARY, INFINITE_BOUNDARY, "
                              "PERIODIC_BOUNDARY");
}

[[nodiscard]] inline viennaps::OxidantType
convertOxidantType(const std::string &value) {
  const auto n = lower(value);
  if (n == "wet" || n == "h2o")
    return viennaps::OxidantType::WET;
  if (n == "dry" || n == "o2")
    return viennaps::OxidantType::DRY;
  throw std::invalid_argument("Unknown oxidant '" + value +
                              "'. Use wet/H2O or dry/O2.");
}

[[nodiscard]] inline viennaps::SiliconOrientation
convertSiliconOrientation(const std::string &value) {
  const auto n = lower(value);
  if (n == "100" || n == "<100>" || n == "si100")
    return viennaps::SiliconOrientation::Si100;
  if (n == "110" || n == "<110>" || n == "si110")
    return viennaps::SiliconOrientation::Si110;
  if (n == "111" || n == "<111>" || n == "si111")
    return viennaps::SiliconOrientation::Si111;
  if (n == "poly" || n == "polysi" || n == "poly-silicon")
    return viennaps::SiliconOrientation::PolySi;
  throw std::invalid_argument("Unknown orientation '" + value +
                              "'. Use 100, 110, 111, or poly.");
}
} // namespace detail

template <typename T> [[nodiscard]] T convert(const std::string &s) {
  if constexpr (std::is_same_v<T, viennals::SpatialSchemeEnum>) {
    return detail::convertSpatialScheme(s);
  } else if constexpr (std::is_same_v<T, viennals::TemporalSchemeEnum>) {
    return detail::convertTemporalScheme(s);
  } else if constexpr (std::is_same_v<T, viennaps::FluxEngineType>) {
    return detail::convertFluxEngineType(s);
  } else if constexpr (std::is_same_v<T, viennahrle::BoundaryType>) {
    return detail::convertBoundaryType(s);
  } else if constexpr (std::is_same_v<T, viennaps::OxidantType>) {
    return detail::convertOxidantType(s);
  } else if constexpr (std::is_same_v<T, viennaps::SiliconOrientation>) {
    return detail::convertSiliconOrientation(s);
  } else {
    throw std::invalid_argument("Unsupported type for conversion.");
  }
}

namespace detail {
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
  case viennals::SpatialSchemeEnum::WENO_3RD_ORDER:
    return "WENO_3RD_ORDER";
  case viennals::SpatialSchemeEnum::WENO_5TH_ORDER:
    return "WENO_5TH_ORDER";
  default:
    throw std::invalid_argument("Unknown discretization scheme.");
  }
}

[[nodiscard]] inline std::string
convertFluxEngineTypeToString(viennaps::FluxEngineType type) {
  switch (type) {
  case viennaps::FluxEngineType::AUTO:
    return "AUTO";
  case viennaps::FluxEngineType::CPU_DISK:
    return "CPU_DISK";
  case viennaps::FluxEngineType::CPU_TRIANGLE:
    return "CPU_TRIANGLE";
  case viennaps::FluxEngineType::GPU_DISK:
    return "GPU_DISK";
  case viennaps::FluxEngineType::GPU_TRIANGLE:
    return "GPU_TRIANGLE";
  case viennaps::FluxEngineType::GPU_LINE:
    return "GPU_LINE";
  default:
    return "UNKNOWN";
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

[[nodiscard]] inline std::string
convertBoundaryConditionToString(viennals::BoundaryConditionEnum scheme) {
  switch (scheme) {
  case viennals::BoundaryConditionEnum::REFLECTIVE_BOUNDARY:
    return "REFLECTIVE_BOUNDARY";
  case viennals::BoundaryConditionEnum::INFINITE_BOUNDARY:
    return "INFINITE_BOUNDARY";
  case viennals::BoundaryConditionEnum::PERIODIC_BOUNDARY:
    return "PERIODIC_BOUNDARY";
  case viennals::BoundaryConditionEnum::POS_INFINITE_BOUNDARY:
    return "POS_INFINITE_BOUNDARY";
  case viennals::BoundaryConditionEnum::NEG_INFINITE_BOUNDARY:
    return "NEG_INFINITE_BOUNDARY";
  default:
    throw std::invalid_argument("Unknown boundary condition.");
  }
}

[[nodiscard]] inline std::string
convertOxidantTypeToString(viennaps::OxidantType type) {
  switch (type) {
  case viennaps::OxidantType::DRY:
    return "DRY";
  case viennaps::OxidantType::WET:
    return "WET";
  default:
    throw std::invalid_argument("Unknown oxidant type.");
  }
}

[[nodiscard]] inline std::string
convertSiliconOrientationToString(viennaps::SiliconOrientation orientation) {
  switch (orientation) {
  case viennaps::SiliconOrientation::Si100:
    return "Si100";
  case viennaps::SiliconOrientation::Si110:
    return "Si110";
  case viennaps::SiliconOrientation::Si111:
    return "Si111";
  case viennaps::SiliconOrientation::PolySi:
    return "PolySi";
  default:
    throw std::invalid_argument("Unknown silicon orientation.");
  }
}

} // namespace detail

template <typename T> [[nodiscard]] std::string toString(const T &value) {
  if constexpr (std::is_same_v<T, bool>)
    return value ? "true" : "false";
  else if constexpr (std::is_same_v<T, viennals::SpatialSchemeEnum>)
    return detail::convertSpatialSchemeToString(value);
  else if constexpr (std::is_same_v<T, viennals::TemporalSchemeEnum>)
    return detail::convertTemporalSchemeToString(value);
  else if constexpr (std::is_same_v<T, viennaps::Material>) {
    return viennaps::MaterialMap::toString(value);
  } else if constexpr (std::is_same_v<T, viennaps::FluxEngineType>) {
    return detail::convertFluxEngineTypeToString(value);
  } else if constexpr (std::is_same_v<T, viennahrle::BoundaryType>) {
    return detail::convertBoundaryConditionToString(
        static_cast<viennals::BoundaryConditionEnum>(value));
  } else if constexpr (std::is_same_v<T, viennaps::OxidantType>) {
    return detail::convertOxidantTypeToString(value);
  } else if constexpr (std::is_same_v<T, viennaps::SiliconOrientation>) {
    return detail::convertSiliconOrientationToString(value);
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
} // namespace viennacore::util
