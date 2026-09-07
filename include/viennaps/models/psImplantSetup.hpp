#pragma once

#include "../psUnits.hpp"
#include "psImplantParams.hpp"
#include "psModelDbLookup.hpp"
#include "psModelNames.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <string>
#include <vector>

namespace viennaps {

inline ImplantDoseControl implantDoseControlFromString(
    std::string value,
    const ImplantDoseControl fallback = ImplantDoseControl::WaferDose) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  if (value == "waferdose" || value == "wafer_dose" || value == "wafer")
    return ImplantDoseControl::WaferDose;
  if (value == "beamdose" || value == "beam_dose" || value == "beam")
    return ImplantDoseControl::BeamDose;
  if (value == "off" || value == "none")
    return ImplantDoseControl::Off;
  return fallback;
}

template <typename NumericType>
inline NumericType lengthUnitInCentimeters(const std::string &lengthUnit) {
  return static_cast<NumericType>(
      units::lengthUnitToCentimeterScale(lengthUnit));
}

struct DopantFields {
  std::string total;
  std::string active;
  std::string damage;
  std::string lastDamage;
  std::string interstitial;
  std::string vacancy;
  std::string cluster;
  std::string beamHits;
};

inline DopantFields dopantFields(const std::string &species) {
  const auto token = model::canonicalSpeciesToken(species);
  const auto prefix = token.empty() ? std::string("dopant") : token;
  return {prefix + "_total",        prefix + "_active",
          prefix + "_damage",       prefix + "_damage_last",
          prefix + "_interstitial", prefix + "_vacancy",
          prefix + "_cluster",      prefix + "_beam_hits"};
}

template <typename NumericType> struct AnalyticImplantRecipe {
  std::string species = "B";
  std::string material = "Si";
  NumericType energyKeV = 0;
  NumericType tiltDeg = NumericType(7);
  NumericType rotationDeg = 0;
  NumericType doseCm2 = 0;
  NumericType screenThickness = 0;

  PearsonIVParameters<NumericType> head;
  NumericType headLateralMu = 0;
  NumericType headLateralSigma = NumericType(5);

  bool useDualPearson = false;
  NumericType headFraction = 0;
  PearsonIVParameters<NumericType> tail;
  NumericType tailLateralMu = 0;
  NumericType tailLateralSigma = NumericType(5);

  NumericType damageProjectedRange = 0;
  NumericType damageVerticalSigma = 1;
  NumericType damageLambda = 0;
  NumericType damageDefectsPerIon = 0;
  NumericType damageLateralSigma = 1;
  NumericType damageLateralDeltaSigma = 0;
};

template <typename NumericType> struct TableImplantRecipe {
  std::string species = "B";
  std::string material = "Si";
  std::string substrateType = "crystalline";
  NumericType energyKeV = 0;
  NumericType tiltDeg = NumericType(7);
  NumericType rotationDeg = 0;
  NumericType doseCm2 = 0;
  NumericType screenThickness = 0;
  NumericType damageLevel = 0;
};

template <typename NumericType, int D, typename ParameterSet>
struct ImplantSetup {
  ParameterSet parameters;
  std::string species;
  std::string material;
  std::string substrateType;
  NumericType energyKeV = 0;
  NumericType tiltDeg = 0;
  NumericType rotationDeg = 0;
  NumericType doseCm2 = 0;
  NumericType lengthUnitInCm = NumericType(1e-7);
  ImplantDoseControl doseControl = ImplantDoseControl::WaferDose;
  std::vector<Material> maskMaterials = {Material::Mask};
  std::vector<Material> screenMaterials = {Material::SiO2};
  DopantFields labels;
  std::string description;
};

template <typename NumericType, int D>
using AnalyticImplantSetup =
    ImplantSetup<NumericType, D, ImplantParams<NumericType, D>>;

template <typename NumericType, int D>
using TableImplantSetup =
    ImplantSetup<NumericType, D, ImplantTableParams<NumericType>>;

template <typename NumericType, int D>
inline AnalyticImplantSetup<NumericType, D> makeAnalyticImplant(
    const AnalyticImplantRecipe<NumericType> &recipe,
    const NumericType lengthUnitInCm = NumericType(1e-7),
    const ImplantDoseControl doseControl = ImplantDoseControl::WaferDose,
    const std::vector<Material> &maskMaterials = {Material::Mask},
    const std::vector<Material> &screenMaterials = {Material::SiO2}) {
  AnalyticImplantSetup<NumericType, D> out;
  out.species = recipe.species;
  out.material = recipe.material;
  out.energyKeV = recipe.energyKeV;
  out.tiltDeg = recipe.tiltDeg;
  out.rotationDeg = recipe.rotationDeg;
  out.doseCm2 = recipe.doseCm2;
  out.lengthUnitInCm = lengthUnitInCm;
  out.doseControl = doseControl;
  out.maskMaterials = maskMaterials;
  out.screenMaterials = screenMaterials;
  out.labels = dopantFields(recipe.species);
  if (recipe.useDualPearson) {
    out.parameters.implantModel =
        SmartPointer<ImplantDualPearsonIV<NumericType, D>>::New(
            recipe.head, recipe.tail, recipe.headFraction, recipe.headLateralMu,
            recipe.headLateralSigma, recipe.tailLateralMu,
            recipe.tailLateralSigma);
  } else {
    out.parameters.implantModel =
        SmartPointer<ImplantPearsonIV<NumericType, D>>::New(
            recipe.head, recipe.headLateralMu, recipe.headLateralSigma);
  }

  std::ostringstream description;
  description << recipe.species << " into " << recipe.material << " at "
              << recipe.energyKeV << " keV, " << recipe.tiltDeg << " deg tilt";
  if (recipe.useDualPearson)
    description << " (dual-Pearson IV, head fraction " << recipe.headFraction
                << ")";
  else
    description << " (Pearson IV)";
  out.description = description.str();
  out.parameters.damageModel =
      SmartPointer<ImplantDamageHobler<NumericType, D>>::New(
          recipe.damageProjectedRange, recipe.damageVerticalSigma,
          recipe.damageLambda, recipe.damageDefectsPerIon,
          recipe.damageLateralSigma, recipe.damageLateralDeltaSigma);
  return out;
}

template <typename NumericType, int D>
inline TableImplantSetup<NumericType, D> makeTableImplant(
    const TableImplantRecipe<NumericType> &recipe,
    const NumericType lengthUnitInCm = NumericType(1e-7),
    const ImplantDoseControl doseControl = ImplantDoseControl::WaferDose,
    const std::vector<Material> &maskMaterials = {Material::Mask},
    const std::vector<Material> &screenMaterials = {Material::SiO2}) {
  TableImplantSetup<NumericType, D> out;
  out.species = recipe.species;
  out.material = recipe.material;
  out.substrateType = recipe.substrateType;
  out.energyKeV = recipe.energyKeV;
  out.tiltDeg = recipe.tiltDeg;
  out.rotationDeg = recipe.rotationDeg;
  out.doseCm2 = recipe.doseCm2;
  out.lengthUnitInCm = lengthUnitInCm;
  out.doseControl = doseControl;
  out.maskMaterials = maskMaterials;
  out.screenMaterials = screenMaterials;
  out.labels = dopantFields(recipe.species);
  out.parameters = modeldb::lookupImplantTable<NumericType, D>(
      recipe.species, recipe.material, recipe.substrateType, recipe.energyKeV,
      recipe.tiltDeg, recipe.rotationDeg, recipe.doseCm2,
      recipe.screenThickness, recipe.damageLevel);

  std::ostringstream description;
  description << recipe.species << " into " << recipe.material << " at "
              << recipe.energyKeV << " keV, " << recipe.tiltDeg
              << " deg tilt (table lookup)";
  out.description = description.str();
  return out;
}

template <typename NumericType, int D, typename ParameterSet>
inline void
applyImplantSetup(IonImplantation<NumericType, D> &implant,
                  const ImplantSetup<NumericType, D, ParameterSet> &config) {
  applyImplantParams(implant, config.parameters);
  implant.setTiltAngle(config.tiltDeg);
  implant.setDose(config.doseCm2);
  implant.setLengthUnit(config.lengthUnitInCm);
  implant.setDoseControl(config.doseControl);
  implant.setMaskMaterials(config.maskMaterials);
  implant.setScreenMaterials(config.screenMaterials);
  implant.setConcentrationLabel(config.labels.total);
  implant.setDamageLabel(config.labels.damage);
  implant.setLastDamageLabel(config.labels.lastDamage);
  implant.setBeamHitsLabel(config.labels.beamHits);
}

} // namespace viennaps
