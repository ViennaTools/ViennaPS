#pragma once

#include "../psModelDb.hpp"
#include "psImplantConstants.hpp"
#include "psImplantDamage.hpp"
#include "psImplantPearson.hpp"
#include <csImplantModel.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace viennaps::tables {

template <typename NumericType> struct ImplantTableEntry {
  std::string species;
  std::string material;
  std::string substrateType = "amorphous";
  std::string modelType = "PearsonIV";

  NumericType energyKeV = 0;
  NumericType tiltDeg = 0;
  NumericType rotationDeg = 0;
  NumericType dosePerCm2 = 0;
  NumericType screenThickness = 0;

  NumericType headFraction = 1;
  NumericType screenDecayLength = 0;
  NumericType damageDecay = 0;
  NumericType tiltDecayDeg = 0;
  NumericType reshapeStrength = 1;

  constants::PearsonIVParameters<NumericType> headParams{};
  NumericType headLateralMu = 0;
  NumericType headLateralSigma = 0;
  std::string headLateralModel = "constant";
  NumericType headLateralScale = 1;
  NumericType headLateralLv = 1;
  NumericType headLateralDeltaSigma = 0;
  NumericType headLateralP1 = 0;
  NumericType headLateralP2 = 0;
  NumericType headLateralP3 = 0;
  NumericType headLateralP4 = 0;
  NumericType headLateralP5 = 0;

  constants::PearsonIVParameters<NumericType> tailParams{};
  NumericType tailLateralMu = 0;
  NumericType tailLateralSigma = 0;
  std::string tailLateralModel = "constant";
  NumericType tailLateralScale = 1;
  NumericType tailLateralLv = 1;
  NumericType tailLateralDeltaSigma = 0;
  NumericType tailLateralP1 = 0;
  NumericType tailLateralP2 = 0;
  NumericType tailLateralP3 = 0;
  NumericType tailLateralP4 = 0;
  NumericType tailLateralP5 = 0;
};

template <typename NumericType> struct ImplantRecipe {
  std::string species = "B";
  std::string material = "Si";
  std::string substrateType = "amorphous";
  std::string preferredModel = "auto";
  std::string tableFileName;
  bool useTableLookup = true;

  NumericType energyKeV = 0;
  NumericType tiltDeg = 0;
  NumericType rotationDeg = 0;
  NumericType dosePerCm2 = 0;
  NumericType screenThickness = 0;
  NumericType damageLevel = 0;

  ImplantTableEntry<NumericType> entry{};
};

template <typename NumericType> struct DamageTableEntry {
  std::string species;
  std::string material;
  NumericType energyKeV = 0;
  NumericType tiltDeg = 0;
  NumericType rotationDeg = 0;
  NumericType dosePerCm2 = 0;
  NumericType screenThickness = 0;

  NumericType projectedRange = 0;
  NumericType verticalSigma = 0;
  NumericType lambda = 0;
  NumericType defectsPerIon = 0;

  NumericType lateralMu = 0;
  NumericType lateralSigma = 0;
  std::string lateralModel = "linear_depth_scale";
  NumericType lateralScale = 1;
  NumericType lateralLv = 1;
  NumericType lateralDeltaSigma = 0;
  NumericType lateralP1 = 0;
  NumericType lateralP2 = 0;
  NumericType lateralP3 = 0;
  NumericType lateralP4 = 0;
  NumericType lateralP5 = 0;
};

template <typename NumericType> struct DamageRecipe {
  std::string species = "B";
  std::string material = "Si";
  std::string tableFileName;
  bool useTableLookup = true;

  NumericType energyKeV = 0;
  NumericType tiltDeg = 0;
  NumericType rotationDeg = 0;
  NumericType dosePerCm2 = 0;
  NumericType screenThickness = 0;

  DamageTableEntry<NumericType> entry{};
};

namespace impl {

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

inline std::vector<std::string> splitCSVLine(const std::string &line) {
  std::vector<std::string> values;
  std::string cell;
  std::stringstream stream(line);
  while (std::getline(stream, cell, ',')) {
    values.push_back(trim(cell));
  }
  if (!line.empty() && line.back() == ',')
    values.emplace_back("");
  return values;
}

template <typename NumericType>
inline NumericType
parseNumeric(const std::unordered_map<std::string, size_t> &idx,
             const std::vector<std::string> &cells, const std::string &key,
             NumericType defaultValue = NumericType(0)) {
  const auto it = idx.find(key);
  if (it == idx.end() || it->second >= cells.size() ||
      cells[it->second].empty())
    return defaultValue;
  return static_cast<NumericType>(std::stod(cells[it->second]));
}

inline std::string
parseString(const std::unordered_map<std::string, size_t> &idx,
            const std::vector<std::string> &cells, const std::string &key,
            const std::string &defaultValue = "") {
  const auto it = idx.find(key);
  if (it == idx.end() || it->second >= cells.size() ||
      cells[it->second].empty())
    return defaultValue;
  return cells[it->second];
}

inline std::string lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return value;
}

inline std::string canonicalSpeciesName(const std::string &value) {
  return viennaps::modeldb::canonicalSpeciesName(value);
}

inline std::string canonicalMaterialName(const std::string &value) {
  return viennaps::modeldb::canonicalMaterialName(value);
}

template <typename NumericType>
inline LateralStraggleModel parseLateralModel(
    const std::string &value,
    LateralStraggleModel defaultValue = LateralStraggleModel::Constant) {
  const auto lowered = lower(value);
  if (lowered.empty() || lowered == "constant")
    return LateralStraggleModel::Constant;
  if (lowered == "exponential_depth_decay")
    return LateralStraggleModel::ExponentialDepthDecay;
  if (lowered == "linear_depth_scale")
    return LateralStraggleModel::LinearDepthScale;
  if (lowered == "log_sum_exp_depth_scale")
    return LateralStraggleModel::LogSumExpDepthScale;
  return defaultValue;
}

template <typename NumericType>
inline LateralStraggleParameters<NumericType>
makeLateralParams(const ImplantTableEntry<NumericType> &entry, bool tail) {
  LateralStraggleParameters<NumericType> params;
  if (tail) {
    params.model = parseLateralModel<NumericType>(entry.tailLateralModel);
    params.mu = entry.tailLateralMu;
    params.sigma = entry.tailLateralSigma;
    params.scale = entry.tailLateralScale;
    params.lv = entry.tailLateralLv;
    params.deltaSigma = entry.tailLateralDeltaSigma;
    params.referenceRange = entry.tailParams.mu;
    params.p1 = entry.tailLateralP1;
    params.p2 = entry.tailLateralP2;
    params.p3 = entry.tailLateralP3;
    params.p4 = entry.tailLateralP4;
    params.p5 = entry.tailLateralP5;
  } else {
    params.model = parseLateralModel<NumericType>(entry.headLateralModel);
    params.mu = entry.headLateralMu;
    params.sigma = entry.headLateralSigma;
    params.scale = entry.headLateralScale;
    params.lv = entry.headLateralLv;
    params.deltaSigma = entry.headLateralDeltaSigma;
    params.referenceRange = entry.headParams.mu;
    params.p1 = entry.headLateralP1;
    params.p2 = entry.headLateralP2;
    params.p3 = entry.headLateralP3;
    params.p4 = entry.headLateralP4;
    params.p5 = entry.headLateralP5;
  }
  return params;
}

template <typename NumericType>
inline NumericType normalizedDistance(NumericType lhs, NumericType rhs,
                                      NumericType scale) {
  return std::abs(lhs - rhs) / std::max(scale, NumericType(1e-9));
}

template <typename NumericType>
inline NumericType
crystallineChannelingScale(const ImplantTableEntry<NumericType> &entry,
                           NumericType tiltDeg, NumericType screenThickness,
                           NumericType damageLevel) {
  NumericType scale = NumericType(1);
  if (entry.screenDecayLength > NumericType(0)) {
    scale *= std::exp(-std::max(screenThickness, NumericType(0)) /
                      entry.screenDecayLength);
  }
  if (entry.damageDecay > NumericType(0)) {
    scale *=
        std::exp(-std::max(damageLevel, NumericType(0)) * entry.damageDecay);
  }
  if (entry.tiltDecayDeg > NumericType(0)) {
    scale *= std::exp(-std::max(tiltDeg, NumericType(0)) / entry.tiltDecayDeg);
  }
  return std::clamp(scale, NumericType(0), NumericType(1));
}

template <typename NumericType>
inline ImplantTableEntry<NumericType>
applyCrystallineReshaping(const ImplantTableEntry<NumericType> &entry,
                          NumericType tiltDeg, NumericType screenThickness,
                          NumericType damageLevel) {
  auto adjusted = entry;
  const auto modelLower = lower(adjusted.modelType);
  if (modelLower != "dualpearson" && modelLower != "dualpearsoniv")
    return adjusted;

  const auto channelingScale =
      crystallineChannelingScale(entry, tiltDeg, screenThickness, damageLevel);
  const auto reshape =
      std::clamp(adjusted.reshapeStrength, NumericType(0), NumericType(1));
  const auto blend =
      NumericType(1) - reshape * (NumericType(1) - channelingScale);

  const auto tailFraction = NumericType(1) - adjusted.headFraction;
  const auto adjustedTailFraction = tailFraction * channelingScale;
  adjusted.headFraction = NumericType(1) - adjustedTailFraction;

  adjusted.tailParams.mu =
      adjusted.headParams.mu +
      blend * (adjusted.tailParams.mu - adjusted.headParams.mu);
  adjusted.tailParams.sigma =
      adjusted.headParams.sigma +
      blend * (adjusted.tailParams.sigma - adjusted.headParams.sigma);
  adjusted.tailParams.beta =
      adjusted.headParams.beta +
      blend * (adjusted.tailParams.beta - adjusted.headParams.beta);
  adjusted.tailParams.gamma =
      adjusted.headParams.gamma +
      blend * (adjusted.tailParams.gamma - adjusted.headParams.gamma);
  adjusted.tailLateralMu =
      adjusted.headLateralMu +
      blend * (adjusted.tailLateralMu - adjusted.headLateralMu);
  adjusted.tailLateralSigma =
      adjusted.headLateralSigma +
      blend * (adjusted.tailLateralSigma - adjusted.headLateralSigma);
  adjusted.tailLateralScale =
      adjusted.headLateralScale +
      blend * (adjusted.tailLateralScale - adjusted.headLateralScale);
  adjusted.tailLateralLv =
      adjusted.headLateralLv +
      blend * (adjusted.tailLateralLv - adjusted.headLateralLv);
  adjusted.tailLateralDeltaSigma =
      adjusted.headLateralDeltaSigma +
      blend * (adjusted.tailLateralDeltaSigma - adjusted.headLateralDeltaSigma);
  adjusted.tailLateralP1 =
      adjusted.headLateralP1 +
      blend * (adjusted.tailLateralP1 - adjusted.headLateralP1);
  adjusted.tailLateralP2 =
      adjusted.headLateralP2 +
      blend * (adjusted.tailLateralP2 - adjusted.headLateralP2);
  adjusted.tailLateralP3 =
      adjusted.headLateralP3 +
      blend * (adjusted.tailLateralP3 - adjusted.headLateralP3);
  adjusted.tailLateralP4 =
      adjusted.headLateralP4 +
      blend * (adjusted.tailLateralP4 - adjusted.headLateralP4);
  adjusted.tailLateralP5 =
      adjusted.headLateralP5 +
      blend * (adjusted.tailLateralP5 - adjusted.headLateralP5);
  return adjusted;
}

template <typename NumericType>
inline ImplantTableEntry<NumericType>
entryFromRecipe(const ImplantRecipe<NumericType> &recipe) {
  auto entry = recipe.entry;
  entry.species = recipe.species;
  entry.material = recipe.material;
  entry.substrateType = recipe.substrateType;
  entry.energyKeV = recipe.energyKeV;
  entry.tiltDeg = recipe.tiltDeg;
  entry.rotationDeg = recipe.rotationDeg;
  entry.dosePerCm2 = recipe.dosePerCm2;
  entry.screenThickness = recipe.screenThickness;
  if (entry.modelType.empty() || entry.modelType == "auto") {
    entry.modelType =
        lower(recipe.preferredModel) != "auto"
            ? recipe.preferredModel
            : (lower(recipe.substrateType) == "crystalline" ? "DualPearsonIV"
                                                            : "PearsonIV");
  }
  if (entry.substrateType.empty())
    entry.substrateType = recipe.substrateType;
  return entry;
}

template <typename NumericType, int D>
inline SmartPointer<ImplantModel<NumericType, D>>
buildModelFromEntry(const ImplantTableEntry<NumericType> &entry) {
  const auto loweredModel = lower(entry.modelType);
  if (loweredModel == "dualpearsoniv" || loweredModel == "dualpearson") {
    return SmartPointer<ImplantDualPearsonIV<NumericType, D>>::New(
        entry.headParams, entry.tailParams, entry.headFraction,
        makeLateralParams(entry, false), makeLateralParams(entry, true));
  }
  return SmartPointer<ImplantPearsonIV<NumericType, D>>::New(
      entry.headParams, makeLateralParams(entry, false));
}

template <typename NumericType>
inline DamageTableEntry<NumericType>
entryFromRecipe(const DamageRecipe<NumericType> &recipe) {
  auto entry = recipe.entry;
  entry.species = recipe.species;
  entry.material = recipe.material;
  entry.energyKeV = recipe.energyKeV;
  entry.tiltDeg = recipe.tiltDeg;
  entry.rotationDeg = recipe.rotationDeg;
  entry.dosePerCm2 = recipe.dosePerCm2;
  entry.screenThickness = recipe.screenThickness;
  return entry;
}

template <typename NumericType>
inline LateralStraggleParameters<NumericType>
makeLateralParams(const DamageTableEntry<NumericType> &entry) {
  LateralStraggleParameters<NumericType> params;
  params.model = parseLateralModel<NumericType>(
      entry.lateralModel, LateralStraggleModel::LinearDepthScale);
  params.mu = entry.lateralMu;
  params.sigma = entry.lateralSigma;
  params.scale = entry.lateralScale;
  params.lv = entry.lateralLv;
  params.deltaSigma = entry.lateralDeltaSigma;
  params.referenceRange = entry.projectedRange;
  params.p1 = entry.lateralP1;
  params.p2 = entry.lateralP2;
  params.p3 = entry.lateralP3;
  params.p4 = entry.lateralP4;
  params.p5 = entry.lateralP5;
  return params;
}

template <typename NumericType, int D>
inline SmartPointer<ImplantModel<NumericType, D>>
buildModelFromEntry(const DamageTableEntry<NumericType> &entry) {
  return SmartPointer<ImplantDamageHobler<NumericType, D>>::New(
      entry.projectedRange, entry.verticalSigma, entry.lambda,
      entry.defectsPerIon, makeLateralParams(entry));
}

} // namespace impl

template <typename NumericType> class ImplantTable {
public:
  ImplantTable() = default;
  explicit ImplantTable(const std::string &fileName) { load(fileName); }

  void load(const std::string &fileName) {
    entries_.clear();
    loadCSV(fileName);
  }

  void loadCSV(const std::string &fileName) {
    entries_.clear();

    std::ifstream file(fileName);
    if (!file.is_open()) {
      throw std::runtime_error("Implant table file could not be opened: " +
                               fileName);
    }

    std::string headerLine;
    while (std::getline(file, headerLine)) {
      headerLine = impl::trim(headerLine);
      if (!headerLine.empty() && headerLine[0] != '#')
        break;
    }
    if (headerLine.empty()) {
      throw std::runtime_error("Implant table file is empty: " + fileName);
    }

    const auto headers = impl::splitCSVLine(headerLine);
    std::unordered_map<std::string, size_t> headerIndex;
    for (size_t i = 0; i < headers.size(); ++i) {
      headerIndex[headers[i]] = i;
    }

    std::string line;
    while (std::getline(file, line)) {
      line = impl::trim(line);
      if (line.empty() || line[0] == '#')
        continue;

      const auto cells = impl::splitCSVLine(line);
      ImplantTableEntry<NumericType> entry;
      entry.species = impl::parseString(headerIndex, cells, "species");
      entry.material = impl::parseString(headerIndex, cells, "material");
      entry.substrateType =
          impl::parseString(headerIndex, cells, "substrateType", "amorphous");
      entry.modelType =
          impl::parseString(headerIndex, cells, "modelType", "PearsonIV");
      entry.energyKeV =
          impl::parseNumeric<NumericType>(headerIndex, cells, "energyKeV");
      entry.tiltDeg =
          impl::parseNumeric<NumericType>(headerIndex, cells, "tiltDeg");
      entry.rotationDeg =
          impl::parseNumeric<NumericType>(headerIndex, cells, "rotationDeg");
      entry.dosePerCm2 =
          impl::parseNumeric<NumericType>(headerIndex, cells, "dosePerCm2");
      entry.screenThickness = impl::parseNumeric<NumericType>(
          headerIndex, cells, "screenThickness");
      entry.headFraction = impl::parseNumeric<NumericType>(
          headerIndex, cells, "headFraction", NumericType(1));
      entry.screenDecayLength = impl::parseNumeric<NumericType>(
          headerIndex, cells, "screenDecayLength", NumericType(0));
      entry.damageDecay = impl::parseNumeric<NumericType>(
          headerIndex, cells, "damageDecay", NumericType(0));
      entry.tiltDecayDeg = impl::parseNumeric<NumericType>(
          headerIndex, cells, "tiltDecayDeg", NumericType(0));
      entry.reshapeStrength = impl::parseNumeric<NumericType>(
          headerIndex, cells, "reshapeStrength", NumericType(1));

      entry.headParams.mu =
          impl::parseNumeric<NumericType>(headerIndex, cells, "headMu");
      entry.headParams.sigma =
          impl::parseNumeric<NumericType>(headerIndex, cells, "headSigma");
      entry.headParams.beta =
          impl::parseNumeric<NumericType>(headerIndex, cells, "headBeta");
      entry.headParams.gamma =
          impl::parseNumeric<NumericType>(headerIndex, cells, "headGamma");
      entry.headLateralMu =
          impl::parseNumeric<NumericType>(headerIndex, cells, "headLateralMu");
      entry.headLateralSigma = impl::parseNumeric<NumericType>(
          headerIndex, cells, "headLateralSigma");
      entry.headLateralModel =
          impl::parseString(headerIndex, cells, "headLateralModel", "constant");
      entry.headLateralScale = impl::parseNumeric<NumericType>(
          headerIndex, cells, "headLateralScale", NumericType(1));
      entry.headLateralLv = impl::parseNumeric<NumericType>(
          headerIndex, cells, "headLateralLv", NumericType(1));
      entry.headLateralDeltaSigma = impl::parseNumeric<NumericType>(
          headerIndex, cells, "headLateralDeltaSigma", NumericType(0));
      entry.headLateralP1 =
          impl::parseNumeric<NumericType>(headerIndex, cells, "headLateralP1");
      entry.headLateralP2 =
          impl::parseNumeric<NumericType>(headerIndex, cells, "headLateralP2");
      entry.headLateralP3 =
          impl::parseNumeric<NumericType>(headerIndex, cells, "headLateralP3");
      entry.headLateralP4 =
          impl::parseNumeric<NumericType>(headerIndex, cells, "headLateralP4");
      entry.headLateralP5 =
          impl::parseNumeric<NumericType>(headerIndex, cells, "headLateralP5");

      entry.tailParams.mu =
          impl::parseNumeric<NumericType>(headerIndex, cells, "tailMu");
      entry.tailParams.sigma =
          impl::parseNumeric<NumericType>(headerIndex, cells, "tailSigma");
      entry.tailParams.beta =
          impl::parseNumeric<NumericType>(headerIndex, cells, "tailBeta");
      entry.tailParams.gamma =
          impl::parseNumeric<NumericType>(headerIndex, cells, "tailGamma");
      entry.tailLateralMu = impl::parseNumeric<NumericType>(
          headerIndex, cells, "tailLateralMu", entry.headLateralMu);
      entry.tailLateralSigma = impl::parseNumeric<NumericType>(
          headerIndex, cells, "tailLateralSigma", entry.headLateralSigma);
      entry.tailLateralModel = impl::parseString(
          headerIndex, cells, "tailLateralModel", entry.headLateralModel);
      entry.tailLateralScale = impl::parseNumeric<NumericType>(
          headerIndex, cells, "tailLateralScale", entry.headLateralScale);
      entry.tailLateralLv = impl::parseNumeric<NumericType>(
          headerIndex, cells, "tailLateralLv", entry.headLateralLv);
      entry.tailLateralDeltaSigma = impl::parseNumeric<NumericType>(
          headerIndex, cells, "tailLateralDeltaSigma",
          entry.headLateralDeltaSigma);
      entry.tailLateralP1 = impl::parseNumeric<NumericType>(
          headerIndex, cells, "tailLateralP1", entry.headLateralP1);
      entry.tailLateralP2 = impl::parseNumeric<NumericType>(
          headerIndex, cells, "tailLateralP2", entry.headLateralP2);
      entry.tailLateralP3 = impl::parseNumeric<NumericType>(
          headerIndex, cells, "tailLateralP3", entry.headLateralP3);
      entry.tailLateralP4 = impl::parseNumeric<NumericType>(
          headerIndex, cells, "tailLateralP4", entry.headLateralP4);
      entry.tailLateralP5 = impl::parseNumeric<NumericType>(
          headerIndex, cells, "tailLateralP5", entry.headLateralP5);

      if (!entry.species.empty() && !entry.material.empty())
        entries_.push_back(entry);
    }
  }

  const std::vector<ImplantTableEntry<NumericType>> &getEntries() const {
    return entries_;
  }

  ImplantTableEntry<NumericType>
  lookup(const std::string &species, const std::string &material,
         const std::string &substrateType, NumericType energyKeV,
         NumericType tiltDeg, NumericType rotationDeg,
         NumericType dosePerCm2 = NumericType(0),
         NumericType screenThickness = NumericType(0),
         const std::string &preferredModel = "auto") const {
    std::vector<const ImplantTableEntry<NumericType> *> candidates;
    const auto speciesLower = impl::canonicalSpeciesName(species);
    const auto materialLower = impl::canonicalMaterialName(material);
    const auto substrateLower = impl::lower(substrateType);
    const auto preferredLower = impl::lower(preferredModel);

    for (const auto &entry : entries_) {
      if (!entry.species.empty() &&
          impl::canonicalSpeciesName(entry.species) != speciesLower)
        continue;
      if (!entry.material.empty() &&
          impl::canonicalMaterialName(entry.material) != materialLower)
        continue;
      if (impl::lower(entry.substrateType) != substrateLower)
        continue;
      if (preferredLower != "auto" &&
          impl::lower(entry.modelType) != preferredLower)
        continue;
      candidates.push_back(&entry);
    }

    if (candidates.empty()) {
      throw std::runtime_error("No implant table entries found for " + species +
                               " in " + material + " (" + substrateType + ").");
    }

    auto restrictToLocalGrid =
        [](std::vector<const ImplantTableEntry<NumericType> *> &entries,
           NumericType target, auto accessor) {
          constexpr NumericType eps = NumericType(1e-12);
          if (entries.size() <= 1)
            return;

          bool hasExact = false;
          NumericType lower = -std::numeric_limits<NumericType>::infinity();
          NumericType upper = std::numeric_limits<NumericType>::infinity();
          for (const auto *entry : entries) {
            const auto value = accessor(*entry);
            if (std::abs(value - target) <= eps) {
              hasExact = true;
              lower = target;
              upper = target;
              break;
            }
            if (value < target && value > lower)
              lower = value;
            if (value > target && value < upper)
              upper = value;
          }

          std::vector<const ImplantTableEntry<NumericType> *> filtered;
          filtered.reserve(entries.size());
          for (const auto *entry : entries) {
            const auto value = accessor(*entry);
            if (hasExact) {
              if (std::abs(value - target) <= eps)
                filtered.push_back(entry);
              continue;
            }

            const bool useLower =
                std::isfinite(lower) && std::abs(value - lower) <= eps;
            const bool useUpper =
                std::isfinite(upper) && std::abs(value - upper) <= eps;
            if (useLower || useUpper)
              filtered.push_back(entry);
          }

          if (!filtered.empty())
            entries = std::move(filtered);
        };

    restrictToLocalGrid(candidates, energyKeV,
                        [](const auto &entry) { return entry.energyKeV; });
    restrictToLocalGrid(candidates, tiltDeg,
                        [](const auto &entry) { return entry.tiltDeg; });
    restrictToLocalGrid(candidates, rotationDeg,
                        [](const auto &entry) { return entry.rotationDeg; });
    restrictToLocalGrid(candidates, dosePerCm2,
                        [](const auto &entry) { return entry.dosePerCm2; });
    restrictToLocalGrid(candidates, screenThickness, [](const auto &entry) {
      return entry.screenThickness;
    });

    if (candidates.size() == 1)
      return *candidates.front();

    NumericType totalWeight = 0;
    ImplantTableEntry<NumericType> result = *candidates.front();
    result.headFraction = 0;
    result.dosePerCm2 = 0;
    result.screenThickness = 0;
    result.screenDecayLength = 0;
    result.damageDecay = 0;
    result.tiltDecayDeg = 0;
    result.reshapeStrength = 0;
    result.headParams = {};
    result.headLateralMu = 0;
    result.headLateralSigma = 0;
    result.headLateralScale = 0;
    result.headLateralLv = 0;
    result.headLateralDeltaSigma = 0;
    result.headLateralP1 = 0;
    result.headLateralP2 = 0;
    result.headLateralP3 = 0;
    result.headLateralP4 = 0;
    result.headLateralP5 = 0;
    result.tailParams = {};
    result.tailLateralMu = 0;
    result.tailLateralSigma = 0;
    result.tailLateralScale = 0;
    result.tailLateralLv = 0;
    result.tailLateralDeltaSigma = 0;
    result.tailLateralP1 = 0;
    result.tailLateralP2 = 0;
    result.tailLateralP3 = 0;
    result.tailLateralP4 = 0;
    result.tailLateralP5 = 0;

    for (const auto *entry : candidates) {
      const auto distance =
          impl::normalizedDistance(entry->energyKeV, energyKeV,
                                   std::max(energyKeV, NumericType(1))) +
          impl::normalizedDistance(entry->tiltDeg, tiltDeg, NumericType(10)) +
          impl::normalizedDistance(entry->rotationDeg, rotationDeg,
                                   NumericType(45)) +
          impl::normalizedDistance(entry->dosePerCm2, dosePerCm2,
                                   std::max(dosePerCm2, NumericType(1e10))) +
          impl::normalizedDistance(entry->screenThickness, screenThickness,
                                   NumericType(10));

      if (distance <= NumericType(1e-12))
        return *entry;

      const auto weight = NumericType(1) / distance;
      totalWeight += weight;

      result.headFraction += weight * entry->headFraction;
      result.dosePerCm2 += weight * entry->dosePerCm2;
      result.screenThickness += weight * entry->screenThickness;
      result.screenDecayLength += weight * entry->screenDecayLength;
      result.damageDecay += weight * entry->damageDecay;
      result.tiltDecayDeg += weight * entry->tiltDecayDeg;
      result.reshapeStrength += weight * entry->reshapeStrength;
      result.headParams.mu += weight * entry->headParams.mu;
      result.headParams.sigma += weight * entry->headParams.sigma;
      result.headParams.beta += weight * entry->headParams.beta;
      result.headParams.gamma += weight * entry->headParams.gamma;
      result.headLateralMu += weight * entry->headLateralMu;
      result.headLateralSigma += weight * entry->headLateralSigma;
      result.headLateralScale += weight * entry->headLateralScale;
      result.headLateralLv += weight * entry->headLateralLv;
      result.headLateralDeltaSigma += weight * entry->headLateralDeltaSigma;
      result.headLateralP1 += weight * entry->headLateralP1;
      result.headLateralP2 += weight * entry->headLateralP2;
      result.headLateralP3 += weight * entry->headLateralP3;
      result.headLateralP4 += weight * entry->headLateralP4;
      result.headLateralP5 += weight * entry->headLateralP5;

      result.tailParams.mu += weight * entry->tailParams.mu;
      result.tailParams.sigma += weight * entry->tailParams.sigma;
      result.tailParams.beta += weight * entry->tailParams.beta;
      result.tailParams.gamma += weight * entry->tailParams.gamma;
      result.tailLateralMu += weight * entry->tailLateralMu;
      result.tailLateralSigma += weight * entry->tailLateralSigma;
      result.tailLateralScale += weight * entry->tailLateralScale;
      result.tailLateralLv += weight * entry->tailLateralLv;
      result.tailLateralDeltaSigma += weight * entry->tailLateralDeltaSigma;
      result.tailLateralP1 += weight * entry->tailLateralP1;
      result.tailLateralP2 += weight * entry->tailLateralP2;
      result.tailLateralP3 += weight * entry->tailLateralP3;
      result.tailLateralP4 += weight * entry->tailLateralP4;
      result.tailLateralP5 += weight * entry->tailLateralP5;
    }

    if (totalWeight <= NumericType(0))
      return *candidates.front();

    result.headFraction /= totalWeight;
    result.dosePerCm2 /= totalWeight;
    result.screenThickness /= totalWeight;
    result.screenDecayLength /= totalWeight;
    result.damageDecay /= totalWeight;
    result.tiltDecayDeg /= totalWeight;
    result.reshapeStrength /= totalWeight;
    result.headParams.mu /= totalWeight;
    result.headParams.sigma /= totalWeight;
    result.headParams.beta /= totalWeight;
    result.headParams.gamma /= totalWeight;
    result.headLateralMu /= totalWeight;
    result.headLateralSigma /= totalWeight;
    result.headLateralScale /= totalWeight;
    result.headLateralLv /= totalWeight;
    result.headLateralDeltaSigma /= totalWeight;
    result.headLateralP1 /= totalWeight;
    result.headLateralP2 /= totalWeight;
    result.headLateralP3 /= totalWeight;
    result.headLateralP4 /= totalWeight;
    result.headLateralP5 /= totalWeight;
    result.tailParams.mu /= totalWeight;
    result.tailParams.sigma /= totalWeight;
    result.tailParams.beta /= totalWeight;
    result.tailParams.gamma /= totalWeight;
    result.tailLateralMu /= totalWeight;
    result.tailLateralSigma /= totalWeight;
    result.tailLateralScale /= totalWeight;
    result.tailLateralLv /= totalWeight;
    result.tailLateralDeltaSigma /= totalWeight;
    result.tailLateralP1 /= totalWeight;
    result.tailLateralP2 /= totalWeight;
    result.tailLateralP3 /= totalWeight;
    result.tailLateralP4 /= totalWeight;
    result.tailLateralP5 /= totalWeight;
    return result;
  }

private:
  std::vector<ImplantTableEntry<NumericType>> entries_;
};

template <typename NumericType> class DamageTable {
public:
  DamageTable() = default;
  explicit DamageTable(const std::string &fileName) { load(fileName); }

  void load(const std::string &fileName) {
    entries_.clear();
    loadCSV(fileName);
  }

  void loadCSV(const std::string &fileName) {
    entries_.clear();
    std::ifstream file(fileName);
    if (!file.is_open()) {
      throw std::runtime_error("Damage table file could not be opened: " +
                               fileName);
    }

    std::string headerLine;
    while (std::getline(file, headerLine)) {
      headerLine = impl::trim(headerLine);
      if (!headerLine.empty() && headerLine[0] != '#')
        break;
    }
    if (headerLine.empty()) {
      throw std::runtime_error("Damage table file is empty: " + fileName);
    }

    const auto headers = impl::splitCSVLine(headerLine);
    std::unordered_map<std::string, size_t> headerIndex;
    for (size_t i = 0; i < headers.size(); ++i) {
      headerIndex[headers[i]] = i;
    }

    std::string line;
    while (std::getline(file, line)) {
      line = impl::trim(line);
      if (line.empty() || line[0] == '#')
        continue;

      const auto cells = impl::splitCSVLine(line);
      DamageTableEntry<NumericType> entry;
      entry.species = impl::parseString(headerIndex, cells, "species");
      entry.material = impl::parseString(headerIndex, cells, "material");
      entry.energyKeV =
          impl::parseNumeric<NumericType>(headerIndex, cells, "energyKeV");
      entry.tiltDeg =
          impl::parseNumeric<NumericType>(headerIndex, cells, "tiltDeg");
      entry.rotationDeg =
          impl::parseNumeric<NumericType>(headerIndex, cells, "rotationDeg");
      entry.dosePerCm2 =
          impl::parseNumeric<NumericType>(headerIndex, cells, "dosePerCm2");
      entry.screenThickness = impl::parseNumeric<NumericType>(
          headerIndex, cells, "screenThickness");
      entry.projectedRange =
          impl::parseNumeric<NumericType>(headerIndex, cells, "projectedRange");
      entry.verticalSigma =
          impl::parseNumeric<NumericType>(headerIndex, cells, "verticalSigma");
      entry.lambda =
          impl::parseNumeric<NumericType>(headerIndex, cells, "lambda");
      entry.defectsPerIon =
          impl::parseNumeric<NumericType>(headerIndex, cells, "defectsPerIon");
      entry.lateralMu =
          impl::parseNumeric<NumericType>(headerIndex, cells, "lateralMu");
      entry.lateralSigma =
          impl::parseNumeric<NumericType>(headerIndex, cells, "lateralSigma");
      entry.lateralModel = impl::parseString(headerIndex, cells, "lateralModel",
                                             "linear_depth_scale");
      entry.lateralScale = impl::parseNumeric<NumericType>(
          headerIndex, cells, "lateralScale", NumericType(1));
      entry.lateralLv = impl::parseNumeric<NumericType>(
          headerIndex, cells, "lateralLv", NumericType(1));
      entry.lateralDeltaSigma = impl::parseNumeric<NumericType>(
          headerIndex, cells, "lateralDeltaSigma", NumericType(0));
      entry.lateralP1 =
          impl::parseNumeric<NumericType>(headerIndex, cells, "lateralP1");
      entry.lateralP2 =
          impl::parseNumeric<NumericType>(headerIndex, cells, "lateralP2");
      entry.lateralP3 =
          impl::parseNumeric<NumericType>(headerIndex, cells, "lateralP3");
      entry.lateralP4 =
          impl::parseNumeric<NumericType>(headerIndex, cells, "lateralP4");
      entry.lateralP5 =
          impl::parseNumeric<NumericType>(headerIndex, cells, "lateralP5");

      if (!entry.species.empty() && !entry.material.empty())
        entries_.push_back(entry);
    }
  }

  const std::vector<DamageTableEntry<NumericType>> &getEntries() const {
    return entries_;
  }

  DamageTableEntry<NumericType>
  lookup(const std::string &species, const std::string &material,
         NumericType energyKeV, NumericType tiltDeg, NumericType rotationDeg,
         NumericType dosePerCm2 = NumericType(0),
         NumericType screenThickness = NumericType(0)) const {
    std::vector<const DamageTableEntry<NumericType> *> candidates;
    const auto speciesLower = impl::canonicalSpeciesName(species);
    const auto materialLower = impl::canonicalMaterialName(material);

    for (const auto &entry : entries_) {
      if (!entry.species.empty() &&
          impl::canonicalSpeciesName(entry.species) != speciesLower)
        continue;
      if (!entry.material.empty() &&
          impl::canonicalMaterialName(entry.material) != materialLower)
        continue;
      candidates.push_back(&entry);
    }

    if (candidates.empty()) {
      throw std::runtime_error("No damage table entries found for " + species +
                               " in " + material + ".");
    }

    auto restrictToLocalGrid =
        [](std::vector<const DamageTableEntry<NumericType> *> &entries,
           NumericType target, auto accessor) {
          constexpr NumericType eps = NumericType(1e-12);
          if (entries.size() <= 1)
            return;

          bool hasExact = false;
          NumericType lower = -std::numeric_limits<NumericType>::infinity();
          NumericType upper = std::numeric_limits<NumericType>::infinity();
          for (const auto *entry : entries) {
            const auto value = accessor(*entry);
            if (std::abs(value - target) <= eps) {
              hasExact = true;
              lower = target;
              upper = target;
              break;
            }
            if (value < target && value > lower)
              lower = value;
            if (value > target && value < upper)
              upper = value;
          }

          std::vector<const DamageTableEntry<NumericType> *> filtered;
          filtered.reserve(entries.size());
          for (const auto *entry : entries) {
            const auto value = accessor(*entry);
            if (hasExact) {
              if (std::abs(value - target) <= eps)
                filtered.push_back(entry);
              continue;
            }

            const bool useLower =
                std::isfinite(lower) && std::abs(value - lower) <= eps;
            const bool useUpper =
                std::isfinite(upper) && std::abs(value - upper) <= eps;
            if (useLower || useUpper)
              filtered.push_back(entry);
          }

          if (!filtered.empty())
            entries = std::move(filtered);
        };

    restrictToLocalGrid(candidates, energyKeV,
                        [](const auto &entry) { return entry.energyKeV; });
    restrictToLocalGrid(candidates, tiltDeg,
                        [](const auto &entry) { return entry.tiltDeg; });
    restrictToLocalGrid(candidates, rotationDeg,
                        [](const auto &entry) { return entry.rotationDeg; });
    restrictToLocalGrid(candidates, dosePerCm2,
                        [](const auto &entry) { return entry.dosePerCm2; });
    restrictToLocalGrid(candidates, screenThickness, [](const auto &entry) {
      return entry.screenThickness;
    });

    if (candidates.size() == 1)
      return *candidates.front();

    NumericType totalWeight = 0;
    DamageTableEntry<NumericType> result = *candidates.front();
    result.projectedRange = 0;
    result.verticalSigma = 0;
    result.lateralSigma = 0;
    result.lateralDeltaSigma = 0;
    result.lambda = 0;
    result.defectsPerIon = 0;
    result.dosePerCm2 = 0;
    result.screenThickness = 0;

    for (const auto *entry : candidates) {
      const auto distance =
          impl::normalizedDistance(entry->energyKeV, energyKeV,
                                   std::max(energyKeV, NumericType(1))) +
          impl::normalizedDistance(entry->tiltDeg, tiltDeg, NumericType(10)) +
          impl::normalizedDistance(entry->rotationDeg, rotationDeg,
                                   NumericType(45)) +
          impl::normalizedDistance(entry->dosePerCm2, dosePerCm2,
                                   std::max(dosePerCm2, NumericType(1e10))) +
          impl::normalizedDistance(entry->screenThickness, screenThickness,
                                   NumericType(10));

      if (distance <= NumericType(1e-12))
        return *entry;

      const auto weight = NumericType(1) / distance;
      totalWeight += weight;
      result.projectedRange += weight * entry->projectedRange;
      result.verticalSigma += weight * entry->verticalSigma;
      result.lateralSigma += weight * entry->lateralSigma;
      result.lateralDeltaSigma += weight * entry->lateralDeltaSigma;
      result.lambda += weight * entry->lambda;
      result.defectsPerIon += weight * entry->defectsPerIon;
      result.dosePerCm2 += weight * entry->dosePerCm2;
      result.screenThickness += weight * entry->screenThickness;
    }

    if (totalWeight <= NumericType(0))
      return *candidates.front();

    result.projectedRange /= totalWeight;
    result.verticalSigma /= totalWeight;
    result.lateralSigma /= totalWeight;
    result.lateralDeltaSigma /= totalWeight;
    result.lambda /= totalWeight;
    result.defectsPerIon /= totalWeight;
    result.dosePerCm2 /= totalWeight;
    result.screenThickness /= totalWeight;
    return result;
  }

private:
  std::vector<DamageTableEntry<NumericType>> entries_;
};

template <typename NumericType>
inline ImplantTableEntry<NumericType>
lookupImplantTableEntry(const std::string &fileName, const std::string &species,
                        const std::string &material,
                        const std::string &substrateType, NumericType energyKeV,
                        NumericType tiltDeg, NumericType rotationDeg,
                        NumericType dosePerCm2 = NumericType(0),
                        NumericType screenThickness = NumericType(0),
                        NumericType damageLevel = NumericType(0),
                        const std::string &preferredModel = "auto") {
  if (fileName.empty())
    throw std::runtime_error(
        "Implant table lookup requires an explicit table file path.");
  ImplantTable<NumericType> table(fileName);
  return impl::applyCrystallineReshaping(
      table.lookup(species, material, substrateType, energyKeV, tiltDeg,
                   rotationDeg, dosePerCm2, screenThickness, preferredModel),
      tiltDeg, screenThickness, damageLevel);
}

template <typename NumericType>
inline DamageTableEntry<NumericType>
lookupDamageTableEntry(const std::string &fileName, const std::string &species,
                       const std::string &material, NumericType energyKeV,
                       NumericType tiltDeg, NumericType rotationDeg,
                       NumericType dosePerCm2 = NumericType(0),
                       NumericType screenThickness = NumericType(0)) {
  if (fileName.empty())
    throw std::runtime_error(
        "Damage table lookup requires an explicit table file path.");
  DamageTable<NumericType> table(fileName);
  return table.lookup(species, material, energyKeV, tiltDeg, rotationDeg,
                      dosePerCm2, screenThickness);
}

template <typename NumericType, int D>
class ImplantTableModel final : public ImplantModel<NumericType, D> {
public:
  ImplantTableModel(const std::string &fileName, const std::string &species,
                    const std::string &material,
                    const std::string &substrateType, NumericType energyKeV,
                    NumericType tiltDeg, NumericType rotationDeg,
                    NumericType dosePerCm2 = NumericType(0),
                    NumericType screenThickness = NumericType(0),
                    NumericType damageLevel = NumericType(0),
                    const std::string &preferredModel = "auto") {
    std::string resolvedFileName = fileName;
    if (resolvedFileName.empty())
      throw std::runtime_error(
          "ImplantTableModel requires an explicit table file path. "
          "Resolve model tables in ViennaPS and pass tableFileName/entry "
          "explicitly.");
    selectedEntry_ = lookupImplantTableEntry(
        resolvedFileName, species, material, substrateType, energyKeV, tiltDeg,
        rotationDeg, dosePerCm2, screenThickness, damageLevel, preferredModel);
    model_ = impl::buildModelFromEntry<NumericType, D>(selectedEntry_);
  }

  NumericType getDepthProfile(NumericType depth) override {
    return model_->getDepthProfile(depth);
  }

  NumericType getLateralProfile(NumericType offset,
                                NumericType depth) override {
    return model_->getLateralProfile(offset, depth);
  }

  NumericType getProfile(NumericType depth, NumericType offset) override {
    return model_->getProfile(depth, offset);
  }

  NumericType getMaxDepth() override { return model_->getMaxDepth(); }

  NumericType getMaxLateralRange() override {
    return model_->getMaxLateralRange();
  }

  const ImplantTableEntry<NumericType> &getSelectedEntry() const {
    return selectedEntry_;
  }

private:
  ImplantTableEntry<NumericType> selectedEntry_;
  SmartPointer<ImplantModel<NumericType, D>> model_;
};

template <typename NumericType, int D>
class ImplantRecipeModel final : public ImplantModel<NumericType, D> {
public:
  ImplantRecipeModel(const ImplantRecipe<NumericType> &recipe,
                     const std::string &defaultTableFileName = "")
      : recipe_(recipe) {
    std::string resolvedTableFileName = !recipe.tableFileName.empty()
                                            ? recipe.tableFileName
                                            : defaultTableFileName;

    if (recipe.useTableLookup) {
      if (resolvedTableFileName.empty())
        throw std::runtime_error(
            "ImplantRecipeModel with table lookup requires explicit "
            "recipe.tableFileName/defaultTableFileName. Resolve in ViennaPS.");
      selectedEntry_ = lookupImplantTableEntry(
          resolvedTableFileName, recipe.species, recipe.material,
          recipe.substrateType, recipe.energyKeV, recipe.tiltDeg,
          recipe.rotationDeg, recipe.dosePerCm2, recipe.screenThickness,
          recipe.damageLevel, recipe.preferredModel);
    } else {
      selectedEntry_ = impl::entryFromRecipe(recipe);
      selectedEntry_ = impl::applyCrystallineReshaping(
          selectedEntry_, recipe.tiltDeg, recipe.screenThickness,
          recipe.damageLevel);
    }
    model_ = impl::buildModelFromEntry<NumericType, D>(selectedEntry_);
  }

  NumericType getDepthProfile(NumericType depth) override {
    return model_->getDepthProfile(depth);
  }

  NumericType getLateralProfile(NumericType offset,
                                NumericType depth) override {
    return model_->getLateralProfile(offset, depth);
  }

  NumericType getProfile(NumericType depth, NumericType offset) override {
    return model_->getProfile(depth, offset);
  }

  NumericType getMaxDepth() override { return model_->getMaxDepth(); }

  NumericType getMaxLateralRange() override {
    return model_->getMaxLateralRange();
  }

  const ImplantRecipe<NumericType> &getRecipe() const { return recipe_; }

  const ImplantTableEntry<NumericType> &getSelectedEntry() const {
    return selectedEntry_;
  }

private:
  ImplantRecipe<NumericType> recipe_;
  ImplantTableEntry<NumericType> selectedEntry_;
  SmartPointer<ImplantModel<NumericType, D>> model_;
};

template <typename NumericType, int D>
class DamageTableModel final : public ImplantModel<NumericType, D> {
public:
  DamageTableModel(const std::string &fileName, const std::string &species,
                   const std::string &material, NumericType energyKeV,
                   NumericType tiltDeg, NumericType rotationDeg,
                   NumericType dosePerCm2 = NumericType(0),
                   NumericType screenThickness = NumericType(0)) {
    std::string resolvedFileName = fileName;
    if (resolvedFileName.empty())
      throw std::runtime_error(
          "DamageTableModel requires an explicit table file path. "
          "Resolve model tables in ViennaPS and pass tableFileName/entry "
          "explicitly.");
    selectedEntry_ = lookupDamageTableEntry(resolvedFileName, species, material,
                                            energyKeV, tiltDeg, rotationDeg,
                                            dosePerCm2, screenThickness);
    model_ = impl::buildModelFromEntry<NumericType, D>(selectedEntry_);
  }

  NumericType getDepthProfile(NumericType depth) override {
    return model_->getDepthProfile(depth);
  }

  NumericType getLateralProfile(NumericType offset,
                                NumericType depth) override {
    return model_->getLateralProfile(offset, depth);
  }

  NumericType getProfile(NumericType depth, NumericType offset) override {
    return model_->getProfile(depth, offset);
  }

  NumericType getMaxDepth() override { return model_->getMaxDepth(); }

  NumericType getMaxLateralRange() override {
    return model_->getMaxLateralRange();
  }

  const DamageTableEntry<NumericType> &getSelectedEntry() const {
    return selectedEntry_;
  }

private:
  DamageTableEntry<NumericType> selectedEntry_;
  SmartPointer<ImplantModel<NumericType, D>> model_;
};

template <typename NumericType, int D>
class DamageRecipeModel final : public ImplantModel<NumericType, D> {
public:
  DamageRecipeModel(const DamageRecipe<NumericType> &recipe,
                    const std::string &defaultTableFileName = "")
      : recipe_(recipe) {
    std::string resolvedTableFileName = !recipe.tableFileName.empty()
                                            ? recipe.tableFileName
                                            : defaultTableFileName;

    if (recipe.useTableLookup) {
      if (resolvedTableFileName.empty())
        throw std::runtime_error(
            "DamageRecipeModel with table lookup requires explicit "
            "recipe.tableFileName/defaultTableFileName. Resolve in ViennaPS.");
      selectedEntry_ = lookupDamageTableEntry(
          resolvedTableFileName, recipe.species, recipe.material,
          recipe.energyKeV, recipe.tiltDeg, recipe.rotationDeg,
          recipe.dosePerCm2, recipe.screenThickness);
    } else {
      selectedEntry_ = impl::entryFromRecipe(recipe);
    }

    model_ = impl::buildModelFromEntry<NumericType, D>(selectedEntry_);
  }

  NumericType getDepthProfile(NumericType depth) override {
    return model_->getDepthProfile(depth);
  }

  NumericType getLateralProfile(NumericType offset,
                                NumericType depth) override {
    return model_->getLateralProfile(offset, depth);
  }

  NumericType getProfile(NumericType depth, NumericType offset) override {
    return model_->getProfile(depth, offset);
  }

  NumericType getMaxDepth() override { return model_->getMaxDepth(); }

  NumericType getMaxLateralRange() override {
    return model_->getMaxLateralRange();
  }

  const DamageRecipe<NumericType> &getRecipe() const { return recipe_; }

  const DamageTableEntry<NumericType> &getSelectedEntry() const {
    return selectedEntry_;
  }

private:
  DamageRecipe<NumericType> recipe_;
  DamageTableEntry<NumericType> selectedEntry_;
  SmartPointer<ImplantModel<NumericType, D>> model_;
};

} // namespace viennaps::tables
