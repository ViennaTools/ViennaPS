#pragma once

#include <algorithm>
#include <cctype>
#include <string>

namespace viennaps::model {

inline std::string trim(std::string value) {
  value.erase(value.begin(),
              std::find_if(value.begin(), value.end(),
                           [](unsigned char c) { return !std::isspace(c); }));
  value.erase(std::find_if(value.rbegin(), value.rend(),
                           [](unsigned char c) { return !std::isspace(c); })
                  .base(),
              value.end());
  return value;
}

inline std::string lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return value;
}

inline std::string canonicalSpeciesToken(const std::string &species) {
  const auto s = lower(trim(species));
  if (s == "b" || s == "boron")
    return "B";
  if (s == "p" || s == "phosphorus")
    return "P";
  if (s == "as" || s == "arsenic")
    return "As";
  if (s == "sb" || s == "antimony")
    return "Sb";
  if (s == "in" || s == "indium")
    return "In";
  return trim(species);
}

inline std::string canonicalMaterialToken(const std::string &material) {
  const auto m = lower(trim(material));
  if (m == "si" || m == "silicon")
    return "Si";
  if (m == "ge" || m == "germanium")
    return "Ge";
  return trim(material);
}

inline std::string canonicalSpeciesName(const std::string &value) {
  const auto tok = canonicalSpeciesToken(value);
  if (tok == "B")
    return "boron";
  if (tok == "P")
    return "phosphorus";
  if (tok == "As")
    return "arsenic";
  if (tok == "Sb")
    return "antimony";
  if (tok == "In")
    return "indium";
  const auto lowered = lower(trim(value));
  if (lowered == "bf2")
    return "bf2";
  if (lowered == "ge")
    return "germanium";
  if (lowered == "si")
    return "silicon";
  return lowered;
}

inline std::string canonicalMaterialName(const std::string &value) {
  const auto tok = canonicalMaterialToken(value);
  if (tok == "Si")
    return "silicon";
  if (tok == "Ge")
    return "germanium";
  const auto lowered = lower(trim(value));
  if (lowered == "sio2" || lowered == "oxide")
    return "oxide";
  if (lowered == "si3n4" || lowered == "nitride")
    return "nitride";
  if (lowered == "poly" || lowered == "polysilicon")
    return "polysilicon";
  return lowered;
}

} // namespace viennaps::model
