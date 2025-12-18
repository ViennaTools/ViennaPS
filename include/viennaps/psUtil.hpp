#pragma once

#include <lsAdvect.hpp>
#include <lsDomain.hpp>
#include <rayBoundary.hpp>

#include <regex>
#include <string>
#include <unordered_map>

// Use viennacore here to avoid conflicts with other namespaces
namespace viennacore::util {
[[nodiscard]] inline viennals::DiscretizationSchemeEnum
convertDiscretizationScheme(const std::string &s) {
  if (s == "ENGQUIST_OSHER_1ST_ORDER" || s == "EO_1")
    return viennals::DiscretizationSchemeEnum::ENGQUIST_OSHER_1ST_ORDER;
  if (s == "ENGQUIST_OSHER_2ND_ORDER" || s == "EO_2")
    return viennals::DiscretizationSchemeEnum::ENGQUIST_OSHER_2ND_ORDER;
  if (s == "LAX_FRIEDRICHS_1ST_ORDER" || s == "LF_1")
    return viennals::DiscretizationSchemeEnum::LAX_FRIEDRICHS_1ST_ORDER;
  if (s == "LAX_FRIEDRICHS_2ND_ORDER" || s == "LF_2")
    return viennals::DiscretizationSchemeEnum::LAX_FRIEDRICHS_2ND_ORDER;
  if (s == "LOCAL_LAX_FRIEDRICHS_ANALYTICAL_1ST_ORDER" || s == "LLFA_1")
    return viennals::DiscretizationSchemeEnum::
        LOCAL_LAX_FRIEDRICHS_ANALYTICAL_1ST_ORDER;
  if (s == "LOCAL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER" || s == "LLLF_1")
    return viennals::DiscretizationSchemeEnum::
        LOCAL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER;
  if (s == "LOCAL_LOCAL_LAX_FRIEDRICHS_2ND_ORDER" || s == "LLLF_2")
    return viennals::DiscretizationSchemeEnum::
        LOCAL_LOCAL_LAX_FRIEDRICHS_2ND_ORDER;
  if (s == "LOCAL_LAX_FRIEDRICHS_1ST_ORDER" || s == "LLF_1")
    return viennals::DiscretizationSchemeEnum::LOCAL_LAX_FRIEDRICHS_1ST_ORDER;
  if (s == "LOCAL_LAX_FRIEDRICHS_2ND_ORDER" || s == "LLF_2")
    return viennals::DiscretizationSchemeEnum::LOCAL_LAX_FRIEDRICHS_2ND_ORDER;
  if (s == "STENCIL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER" || s == "SLLF_1")
    return viennals::DiscretizationSchemeEnum::
        STENCIL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER;
  if (s == "WENO_5TH_ORDER" || s == "WENO_5")
    return viennals::DiscretizationSchemeEnum::WENO_5TH_ORDER;
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

[[nodiscard]] inline std::string
convertDiscretizationSchemeToString(viennals::DiscretizationSchemeEnum scheme) {
  switch (scheme) {
  case viennals::DiscretizationSchemeEnum::ENGQUIST_OSHER_1ST_ORDER:
    return "ENGQUIST_OSHER_1ST_ORDER";
  case viennals::DiscretizationSchemeEnum::ENGQUIST_OSHER_2ND_ORDER:
    return "ENGQUIST_OSHER_2ND_ORDER";
  case viennals::DiscretizationSchemeEnum::LAX_FRIEDRICHS_1ST_ORDER:
    return "LAX_FRIEDRICHS_1ST_ORDER";
  case viennals::DiscretizationSchemeEnum::LAX_FRIEDRICHS_2ND_ORDER:
    return "LAX_FRIEDRICHS_2ND_ORDER";
  case viennals::DiscretizationSchemeEnum::
      LOCAL_LAX_FRIEDRICHS_ANALYTICAL_1ST_ORDER:
    return "LOCAL_LAX_FRIEDRICHS_ANALYTICAL_1ST_ORDER";
  case viennals::DiscretizationSchemeEnum::LOCAL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER:
    return "LOCAL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER";
  case viennals::DiscretizationSchemeEnum::LOCAL_LOCAL_LAX_FRIEDRICHS_2ND_ORDER:
    return "LOCAL_LOCAL_LAX_FRIEDRICHS_2ND_ORDER";
  case viennals::DiscretizationSchemeEnum::LOCAL_LAX_FRIEDRICHS_1ST_ORDER:
    return "LOCAL_LAX_FRIEDRICHS_1ST_ORDER";
  case viennals::DiscretizationSchemeEnum::LOCAL_LAX_FRIEDRICHS_2ND_ORDER:
    return "LOCAL_LAX_FRIEDRICHS_2ND_ORDER";
  case viennals::DiscretizationSchemeEnum::
      STENCIL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER:
    return "STENCIL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER";
  case viennals::DiscretizationSchemeEnum::WENO_5TH_ORDER:
    return "WENO_5TH_ORDER";
  default:
    throw std::invalid_argument("Unknown discretization scheme.");
  }
}

[[nodiscard]] inline viennaray::BoundaryCondition convertBoundaryCondition(
    viennals::BoundaryConditionEnum originalBoundaryCondition) {
  switch (originalBoundaryCondition) {
  case viennals::BoundaryConditionEnum::REFLECTIVE_BOUNDARY:
    return viennaray::BoundaryCondition::REFLECTIVE;

  case viennals::BoundaryConditionEnum::INFINITE_BOUNDARY:
    return viennaray::BoundaryCondition::IGNORE;

  case viennals::BoundaryConditionEnum::PERIODIC_BOUNDARY:
    return viennaray::BoundaryCondition::PERIODIC;

  case viennals::BoundaryConditionEnum::POS_INFINITE_BOUNDARY:
  case viennals::BoundaryConditionEnum::NEG_INFINITE_BOUNDARY:
    return viennaray::BoundaryCondition::IGNORE;
  }
  return viennaray::BoundaryCondition::IGNORE;
}

[[nodiscard]] inline std::string metaDataToString(
    const std::unordered_map<std::string, std::vector<double>> &metaData) {
  std::string result;
  for (const auto &item : metaData) {
    result += "\n" + item.first + ": ";
    for (const auto &value : item.second) {
      result += std::to_string(value) + " ";
    }
  }
  return result;
}
}; // namespace viennacore::util
