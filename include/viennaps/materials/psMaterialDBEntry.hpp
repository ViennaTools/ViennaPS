#pragma once

#include "psBuiltInMaterial.hpp"

namespace viennaps::materials {

static constexpr double kNaNDensity = std::numeric_limits<double>::quiet_NaN();
static constexpr uint32_t kNaNColor = 0xcccccc;

struct DBEntry {
  std::string name = "";
  MaterialCategory category = MaterialCategory::Generic;
  bool conductive = false;
  uint32_t colorHex = kNaNColor;

  double density_gcm3 = kNaNDensity; // g/cm^3
  double density_acm3 = kNaNDensity; // atoms/cm^3
};

} // namespace viennaps::materials