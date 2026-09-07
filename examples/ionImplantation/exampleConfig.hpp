#pragma once

#include <models/psAnnealSetup.hpp>
#include <models/psImplantSetup.hpp>

#include <vcUtil.hpp>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace ionimpl {

using RawParameters = std::unordered_map<std::string, std::string>;

inline std::string lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return value;
}

inline std::string trim(std::string value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos)
    return {};
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

inline RawParameters readRawParameters(const std::string &path) {
  RawParameters out;
  std::ifstream file(path);
  std::string line;
  while (std::getline(file, line)) {
    const auto hashPos = line.find('#');
    if (hashPos != std::string::npos)
      line = line.substr(0, hashPos);
    const auto eqPos = line.find('=');
    if (eqPos == std::string::npos)
      continue;
    const auto key = trim(line.substr(0, eqPos));
    const auto value = trim(line.substr(eqPos + 1));
    if (!key.empty())
      out[key] = value;
  }
  return out;
}

template <typename NumericType>
inline std::vector<NumericType> parseNumberList(const std::string &value) {
  std::vector<NumericType> result;
  std::istringstream stream(value);
  std::string token;
  while (std::getline(stream, token, ',')) {
    token = trim(token);
    if (!token.empty())
      result.push_back(static_cast<NumericType>(std::stod(token)));
  }
  return result;
}

inline bool getBool(const viennacore::util::Parameters &params, const char *key,
                    const bool fallback) {
  const auto it = params.m.find(key);
  if (it == params.m.end())
    return fallback;
  const auto value = lower(it->second);
  if (value == "1" || value == "true" || value == "yes" || value == "on")
    return true;
  if (value == "0" || value == "false" || value == "no" || value == "off")
    return false;
  return std::stod(it->second) != 0.0;
}

template <typename NumericType>
inline NumericType getNumber(const viennacore::util::Parameters &params,
                             const char *key, const NumericType fallback) {
  const auto it = params.m.find(key);
  if (it == params.m.end())
    return fallback;
  return static_cast<NumericType>(std::stod(it->second));
}

inline std::string getString(const viennacore::util::Parameters &params,
                             const char *key, const std::string &fallback) {
  const auto it = params.m.find(key);
  return it == params.m.end() ? fallback : it->second;
}

inline bool hasAny(const viennacore::util::Parameters &params,
                   std::initializer_list<const char *> keys) {
  for (const auto *key : keys) {
    if (params.m.count(key))
      return true;
  }
  return false;
}

template <typename NumericType>
inline bool assignNumber(const viennacore::util::Parameters &params,
                         std::initializer_list<const char *> keys,
                         NumericType &target) {
  for (const auto *key : keys) {
    const auto it = params.m.find(key);
    if (it == params.m.end())
      continue;
    target = static_cast<NumericType>(std::stod(it->second));
    return true;
  }
  return false;
}

template <typename NumericType>
inline viennaps::AnnealSchedule<NumericType>
readAnnealSchedule(const RawParameters &rawParams) {
  viennaps::AnnealSchedule<NumericType> out;
  const auto durations = rawParams.find("annealStepDurations");
  if (durations != rawParams.end())
    out.durations = parseNumberList<NumericType>(durations->second);
  const auto temperatures = rawParams.find("annealTemperatures");
  if (temperatures != rawParams.end())
    out.temperatures = parseNumberList<NumericType>(temperatures->second);
  return out;
}

template <typename NumericType, int D>
inline viennaps::AnalyticImplantSetup<NumericType, D>
makeAnalyticImplantSetup(const viennacore::util::Parameters &params,
                         const NumericType screenThickness) {
  viennaps::AnalyticImplantRecipe<NumericType> recipe;
  recipe.species = getString(params, "species", "P");
  recipe.material = getString(params, "material", "Si");
  recipe.energyKeV = static_cast<NumericType>(params.get("energyKeV"));
  recipe.tiltDeg = getNumber<NumericType>(params, "angle", NumericType(7));
  recipe.rotationDeg =
      getNumber<NumericType>(params, "rotationDeg", NumericType(0));
  recipe.doseCm2 = static_cast<NumericType>(params.get("doseCm2"));
  recipe.screenThickness = screenThickness;
  recipe.head.mu = static_cast<NumericType>(params.get("projectedRange"));
  recipe.head.sigma = static_cast<NumericType>(params.get("depthSigma"));
  recipe.head.beta = static_cast<NumericType>(params.get("skewness"));
  recipe.head.gamma = static_cast<NumericType>(params.get("kurtosis"));
  recipe.headLateralMu =
      getNumber<NumericType>(params, "lateralMu", NumericType(0));
  recipe.headLateralSigma =
      getNumber<NumericType>(params, "lateralSigma", NumericType(5));
  recipe.damageProjectedRange =
      static_cast<NumericType>(params.get("damageProjectedRange"));
  recipe.damageVerticalSigma =
      static_cast<NumericType>(params.get("damageVerticalSigma"));
  recipe.damageLambda = static_cast<NumericType>(params.get("damageLambda"));
  recipe.damageDefectsPerIon =
      static_cast<NumericType>(params.get("damageDefectsPerIon"));
  recipe.damageLateralSigma =
      static_cast<NumericType>(params.get("damageLateralSigma"));
  recipe.damageLateralDeltaSigma =
      static_cast<NumericType>(params.get("damageLateralDeltaSigma"));

  recipe.useDualPearson = params.m.count("headFraction") != 0;
  if (recipe.useDualPearson) {
    recipe.tail.mu = getNumber<NumericType>(params, "tailProjectedRange",
                                            recipe.head.mu * NumericType(2.5));
    recipe.tail.sigma = getNumber<NumericType>(
        params, "tailDepthSigma", recipe.head.sigma * NumericType(2.5));
    recipe.tail.beta =
        getNumber<NumericType>(params, "tailSkewness", NumericType(0));
    recipe.tail.gamma =
        getNumber<NumericType>(params, "tailKurtosis", NumericType(3));
    recipe.tailLateralMu =
        getNumber<NumericType>(params, "tailLateralMu", NumericType(0));
    recipe.tailLateralSigma = getNumber<NumericType>(params, "tailLateralSigma",
                                                     recipe.headLateralSigma);
    recipe.headFraction = static_cast<NumericType>(params.get("headFraction"));
  }

  const auto lengthUnitInCm = viennaps::lengthUnitInCentimeters<NumericType>(
      getString(params, "lengthUnit", "nm"));
  const auto doseControl = viennaps::implantDoseControlFromString(
      getString(params, "doseControl", "WaferDose"));
  return viennaps::makeAnalyticImplant<NumericType, D>(recipe, lengthUnitInCm,
                                                       doseControl);
}

template <typename NumericType, int D>
inline viennaps::TableImplantSetup<NumericType, D>
makeTableImplantSetup(const viennacore::util::Parameters &params,
                      const NumericType screenThickness) {
  viennaps::TableImplantRecipe<NumericType> recipe;
  recipe.species = getString(params, "species", "B");
  recipe.material = getString(params, "material", "Si");
  recipe.substrateType = getString(params, "substrateType", "crystalline");
  recipe.energyKeV = static_cast<NumericType>(params.get("energyKeV"));
  recipe.tiltDeg = getNumber<NumericType>(params, "angle", NumericType(7));
  recipe.rotationDeg =
      getNumber<NumericType>(params, "rotationDeg", NumericType(0));
  recipe.doseCm2 = static_cast<NumericType>(params.get("doseCm2"));
  recipe.screenThickness = screenThickness;
  recipe.damageLevel =
      getNumber<NumericType>(params, "damageLevel", NumericType(0));

  const auto lengthUnitInCm = viennaps::lengthUnitInCentimeters<NumericType>(
      getString(params, "lengthUnit", "nm"));
  const auto doseControl = viennaps::implantDoseControlFromString(
      getString(params, "doseControl", "WaferDose"));
  return viennaps::makeTableImplant<NumericType, D>(recipe, lengthUnitInCm,
                                                    doseControl);
}

template <typename NumericType>
inline bool applyAnnealOverrides(const viennacore::util::Parameters &params,
                                 viennaps::AnnealParams<NumericType> &p) {
  bool overridden = false;
  auto set = [&](std::initializer_list<const char *> keys,
                 NumericType &target) {
    const bool changed = assignNumber(params, keys, target);
    overridden = overridden || changed;
    return changed;
  };

  if (set({"annealDiffusionCoefficient", "annealD"}, p.diffusionCoefficient))
    p.useConstantDiffusionCoefficient = true;
  set({"annealD0", "annealD0_nm2_per_s"}, p.dopantD0);
  set({"annealEa", "annealEa_eV"}, p.dopantEa_eV);
  set({"annealDefectSourceHistoryWeight"}, p.defectSourceHistoryWeight);
  set({"annealLastDamageWeight"}, p.defectSourceLastDamageWeight);
  set({"annealInterstitialDiffusivity", "annealDi"}, p.interstitialDiffusivity);
  set({"annealVacancyDiffusivity", "annealDv"}, p.vacancyDiffusivity);

  if (params.m.count("annealDefectEquilibrium")) {
    p.enableDefectEquilibrium =
        getBool(params, "annealDefectEquilibrium", p.enableDefectEquilibrium);
    overridden = true;
  }
  if (set({"annealInterstitialEqC0"}, p.interstitialEqC0))
    p.enableDefectEquilibrium = true;
  if (set({"annealInterstitialEqEa", "annealInterstitialEqEa_eV"},
          p.interstitialEqEa_eV))
    p.enableDefectEquilibrium = true;
  if (set({"annealVacancyEqC0"}, p.vacancyEqC0))
    p.enableDefectEquilibrium = true;
  if (set({"annealVacancyEqEa", "annealVacancyEqEa_eV"}, p.vacancyEqEa_eV))
    p.enableDefectEquilibrium = true;

  set({"annealRecombinationRate"}, p.defectRecombinationRate);
  set({"annealInterstitialSinkRate"}, p.interstitialSinkRate);
  set({"annealVacancySinkRate"}, p.vacancySinkRate);
  set({"annealScoreIFactor", "annealInterstitialFactor"}, p.scoreIFactor);
  set({"annealScoreVFactor", "annealVacancyFactor"}, p.scoreVFactor);
  set({"annealScoreDFactor", "annealDamageFactor"}, p.scoreDFactor);

  if (params.m.count("annealTedFromScoreDFactor")) {
    p.enableTedFromScoreDFactor = getBool(params, "annealTedFromScoreDFactor",
                                          p.enableTedFromScoreDFactor);
    overridden = true;
  }
  set({"annealTedCoefficient"}, p.tedCoefficient);
  set({"annealTedCoefficientScale"}, p.tedCoefficientScale);
  set({"annealTedNormalization"}, p.tedNormalization);

  if (params.m.count("annealSolidActivation")) {
    p.enableSolidActivation =
        getBool(params, "annealSolidActivation", p.enableSolidActivation);
    overridden = true;
  }
  set({"annealSolidSolubilityC0"}, p.solidSolubilityC0);
  set({"annealSolidSolubilityEa", "annealSolidSolubilityEa_eV"},
      p.solidSolubilityEa_eV);

  if (params.m.count("annealDefectClustering")) {
    p.enableDefectClustering =
        getBool(params, "annealDefectClustering", p.enableDefectClustering);
    overridden = true;
  }
  if (set({"annealClusterKfi"}, p.clusterKfi))
    p.enableDefectClustering = true;
  if (set({"annealClusterKfc"}, p.clusterKfc))
    p.enableDefectClustering = true;
  if (set({"annealClusterKr"}, p.clusterKr))
    p.enableDefectClustering = true;
  if (set({"annealClusterInitFraction"}, p.clusterInitFraction))
    p.enableDefectClustering = true;

  return overridden;
}

template <typename NumericType, typename ImplantSetupT>
inline viennaps::AnnealSetup<NumericType>
makeAnnealSetup(const viennacore::util::Parameters &params,
                const viennaps::AnnealSchedule<NumericType> &schedule,
                const ImplantSetupT &implantSetup,
                const NumericType peakTemperatureK,
                const std::vector<viennaps::Material> &diffusionMaterials =
                    {viennaps::Material::Si},
                const std::vector<viennaps::Material> &blockingMaterials =
                    {viennaps::Material::Mask, viennaps::Material::SiO2},
                const bool defaultUseModelDb = true) {
  bool useModelDb = defaultUseModelDb;
  if (params.m.count("annealParameterSource")) {
    const auto source = lower(params.m.at("annealParameterSource"));
    if (source == "manual" || source == "config" || source == "user")
      useModelDb = false;
    else if (source == "modeldb" || source == "model_db" || source == "table")
      useModelDb = true;
  }
  if (params.m.count("annealUseModelDb"))
    useModelDb = getBool(params, "annealUseModelDb", useModelDb);

  viennaps::AnnealSetup<NumericType> out;
  out.schedule = schedule;
  out.peakTemperatureK = peakTemperatureK;
  out.labels = implantSetup.labels;
  out.diffusionMaterials = diffusionMaterials;
  out.blockingMaterials = blockingMaterials;
  out.substrateMaterial = diffusionMaterials.empty()
                              ? viennaps::Material::Si
                              : diffusionMaterials.front();
  if (useModelDb) {
    out.model = viennaps::lookupAnneal<NumericType>(
        implantSetup, schedule, peakTemperatureK, out.substrateMaterial);
  } else {
    out.model = viennaps::manualAnneal(viennaps::AnnealParams<NumericType>{},
                                       "manual config");
  }

  const bool overridden = applyAnnealOverrides(params, out.model.parameters);
  if (useModelDb && overridden)
    out.model.source += " + config overrides";
  if (!useModelDb && !hasAny(params, {"annealDiffusionCoefficient", "annealD",
                                      "annealD0", "annealD0_nm2_per_s"})) {
    throw std::runtime_error(
        "Manual anneal configuration requires annealDiffusionCoefficient "
        "or annealD0/annealEa.");
  }

  out.duration =
      getNumber<NumericType>(params, "annealDuration", NumericType(5));
  out.mode = viennaps::annealModeFromString(
      getString(params, "annealMode", "implicit"),
      viennaps::AnnealMode::GaussSeidel);
  out.implicitMaxIterations = static_cast<int>(getNumber<NumericType>(
      params, "annealImplicitMaxIterations", NumericType(400)));
  out.implicitTolerance = getNumber<NumericType>(
      params, "annealImplicitTolerance", NumericType(1e-6));
  out.implicitRelaxation = getNumber<NumericType>(
      params, "annealImplicitRelaxation", NumericType(1));
  out.defectCoupling = getBool(params, "annealDefectCoupling", true);
  return out;
}

} // namespace ionimpl
