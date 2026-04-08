#pragma once

#include "../psUtil.hpp"
#include "psBuiltInMaterial.hpp"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>

#include <nlohmann/json.hpp>

namespace viennaps::materials {

static constexpr double kNaNDensity = std::numeric_limits<double>::quiet_NaN();

struct DBEntry {
  MaterialCategory category = MaterialCategory::Generic;

  double density_gcm3 = kNaNDensity; // g/cm^3
  double density_acm3 = kNaNDensity; // atoms/cm^3

  // --- util ---
  using Json = nlohmann::json;

  Json serialize() const {
    Json result;
    result["category"] = util::convertMaterialCategoryToString(category);

    result["density_gcm3"] =
        std::isnan(density_gcm3) ? Json(nullptr) : Json(density_gcm3);
    result["density_acm3"] =
        std::isnan(density_acm3) ? Json(nullptr) : Json(density_acm3);

    return result;
  }

  static std::optional<double> getOptionalDouble(const Json &node,
                                                 const char *field,
                                                 const std::string &path) {
    if (!node.contains(field) || node[field].is_null()) {
      return std::nullopt;
    }
    if (!node[field].is_number()) {
      throw std::runtime_error("Expected numeric field '" + std::string(field) +
                               "' at " + path + ".");
    }
    return node[field].get<double>();
  }

  static DBEntry deserializeEntry(const Json &node,
                                  const std::string &materialName) {
    if (!node.is_object()) {
      throw std::runtime_error("Entry for material '" + materialName +
                               "' must be a JSON object.");
    }

    const std::string path = "materials." + materialName;
    DBEntry entry;

    if (node.contains("category")) {
      if (!node["category"].is_string()) {
        throw std::runtime_error("Expected string field 'category' at " + path +
                                 ".");
      }
      entry.category = util::convertMaterialCategoryFromString(
          node["category"].get<std::string>());
    }

    if (const auto density =
            getOptionalDouble(node, "density_gcm3", path + ".density_gcm3");
        density.has_value()) {
      entry.density_gcm3 = density.value();
    }
    if (const auto density =
            getOptionalDouble(node, "density_acm3", path + ".density_acm3");
        density.has_value()) {
      entry.density_acm3 = density.value();
    }

    return entry;
  }
};

} // namespace viennaps::materials