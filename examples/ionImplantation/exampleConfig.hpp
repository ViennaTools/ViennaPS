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
readAnnealSchedule(const viennaps::util::Parameters &params) {
  viennaps::AnnealSchedule<NumericType> out;
  if (params.contains("annealStepDurations"))
    out.durations = params.get<std::vector<NumericType>>("annealStepDurations");
  if (params.contains("annealTemperatures"))
    out.temperatures =
        params.get<std::vector<NumericType>>("annealTemperatures");
  return out;
}

template <typename NumericType, int D>
inline viennaps::AnalyticImplantSetup<NumericType, D>
makeAnalyticImplantSetup(const viennacore::util::Parameters &params,
                         const NumericType screenThickness) {
  viennaps::AnalyticImplantRecipe<NumericType> recipe;
  recipe.species = params.get<std::string>("species", "P");
  recipe.material = params.get<std::string>("material", "Si");
  recipe.energyKeV = params.get<NumericType>("energyKeV");
  recipe.tiltDeg = params.get<NumericType>("angle", NumericType(7));
  recipe.rotationDeg = params.get<NumericType>("rotationDeg", NumericType(0));
  recipe.doseCm2 = params.get<NumericType>("doseCm2");
  recipe.screenThickness = screenThickness;
  recipe.head.mu = params.get<NumericType>("projectedRange");
  recipe.head.sigma = params.get<NumericType>("depthSigma");
  recipe.head.beta = params.get<NumericType>("skewness");
  recipe.head.gamma = params.get<NumericType>("kurtosis");
  recipe.headLateralMu = params.get<NumericType>("lateralMu", NumericType(0));
  recipe.headLateralSigma =
      params.get<NumericType>("lateralSigma", NumericType(5));
  recipe.damageProjectedRange = params.get<NumericType>("damageProjectedRange");
  recipe.damageVerticalSigma = params.get<NumericType>("damageVerticalSigma");
  recipe.damageLambda = params.get<NumericType>("damageLambda");
  recipe.damageDefectsPerIon = params.get<NumericType>("damageDefectsPerIon");
  recipe.damageLateralSigma = params.get<NumericType>("damageLateralSigma");
  recipe.damageLateralDeltaSigma =
      params.get<NumericType>("damageLateralDeltaSigma");

  recipe.useDualPearson = params.m.count("headFraction") != 0;
  if (recipe.useDualPearson) {
    recipe.tail.mu = params.get<NumericType>("tailProjectedRange",
                                             recipe.head.mu * NumericType(2.5));
    recipe.tail.sigma = params.get<NumericType>(
        "tailDepthSigma", recipe.head.sigma * NumericType(2.5));
    recipe.tail.beta = params.get<NumericType>("tailSkewness", NumericType(0));
    recipe.tail.gamma = params.get<NumericType>("tailKurtosis", NumericType(3));
    recipe.tailLateralMu =
        params.get<NumericType>("tailLateralMu", NumericType(0));
    recipe.tailLateralSigma =
        params.get<NumericType>("tailLateralSigma", recipe.headLateralSigma);
    recipe.headFraction = params.get<NumericType>("headFraction");
  }

  const auto lengthUnitInCm = viennaps::lengthUnitInCentimeters<NumericType>(
      params.get<std::string>("lengthUnit", "nm"));
  const auto doseControl = viennaps::implantDoseControlFromString(
      params.get<std::string>("doseControl", "WaferDose"));
  return viennaps::makeAnalyticImplant<NumericType, D>(recipe, lengthUnitInCm,
                                                       doseControl);
}

template <typename NumericType, int D>
inline viennaps::TableImplantSetup<NumericType, D>
makeTableImplantSetup(const viennacore::util::Parameters &params,
                      const NumericType screenThickness) {
  viennaps::TableImplantRecipe<NumericType> recipe;
  recipe.species = params.get<std::string>("species", "B");
  recipe.material = params.get<std::string>("material", "Si");
  recipe.substrateType =
      params.get<std::string>("substrateType", "crystalline");
  recipe.energyKeV = params.get<NumericType>("energyKeV");
  recipe.tiltDeg = params.get<NumericType>("angle", NumericType(7));
  recipe.rotationDeg = params.get<NumericType>("rotationDeg", NumericType(0));
  recipe.doseCm2 = params.get<NumericType>("doseCm2");
  recipe.screenThickness = screenThickness;
  recipe.damageLevel = params.get<NumericType>("damageLevel", NumericType(0));

  const auto lengthUnitInCm = viennaps::lengthUnitInCentimeters<NumericType>(
      params.get<std::string>("lengthUnit", "nm"));
  const auto doseControl = viennaps::implantDoseControlFromString(
      params.get<std::string>("doseControl", "WaferDose"));
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

  if (params.contains("annealDefectEquilibrium")) {
    p.enableDefectEquilibrium =
        params.get<bool>("annealDefectEquilibrium", p.enableDefectEquilibrium);
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

  if (params.contains("annealTedFromScoreDFactor")) {
    p.enableTedFromScoreDFactor = params.get<bool>("annealTedFromScoreDFactor",
                                                   p.enableTedFromScoreDFactor);
    overridden = true;
  }
  set({"annealTedCoefficient"}, p.tedCoefficient);
  set({"annealTedCoefficientScale"}, p.tedCoefficientScale);
  set({"annealTedNormalization"}, p.tedNormalization);

  if (params.contains("annealSolidActivation")) {
    p.enableSolidActivation =
        params.get<bool>("annealSolidActivation", p.enableSolidActivation);
    overridden = true;
  }
  set({"annealSolidSolubilityC0"}, p.solidSolubilityC0);
  set({"annealSolidSolubilityEa", "annealSolidSolubilityEa_eV"},
      p.solidSolubilityEa_eV);

  if (params.contains("annealDefectClustering")) {
    p.enableDefectClustering =
        params.get<bool>("annealDefectClustering", p.enableDefectClustering);
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
  if (params.contains("annealParameterSource")) {
    const auto source = viennaps::util::detail::lower(
        params.get<std::string>("annealParameterSource"));
    if (source == "manual" || source == "config" || source == "user")
      useModelDb = false;
    else if (source == "modeldb" || source == "model_db" || source == "table")
      useModelDb = true;
  }
  if (params.contains("annealUseModelDb"))
    useModelDb = params.get<bool>("annealUseModelDb", useModelDb);

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
  if (!useModelDb &&
      !params.containsAny({"annealDiffusionCoefficient", "annealD", "annealD0",
                           "annealD0_nm2_per_s"})) {
    throw std::runtime_error(
        "Manual anneal configuration requires annealDiffusionCoefficient "
        "or annealD0/annealEa.");
  }

  out.duration = params.get<NumericType>("annealDuration", NumericType(5));
  out.mode = viennaps::annealModeFromString(
      params.get<std::string>("annealMode", "implicit"),
      viennaps::AnnealMode::GaussSeidel);
  out.implicitMaxIterations =
      params.get<int>("annealImplicitMaxIterations", 400);
  out.implicitTolerance =
      params.get<NumericType>("annealImplicitTolerance", NumericType(1e-6));
  out.implicitRelaxation =
      params.get<NumericType>("annealImplicitRelaxation", NumericType(1));
  out.defectCoupling = params.get<bool>("annealDefectCoupling", true);
  return out;
}

} // namespace ionimpl
