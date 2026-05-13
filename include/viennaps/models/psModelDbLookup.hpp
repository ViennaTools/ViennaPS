#pragma once

#include "psAnneal.hpp"
#include "psAnnealParams.hpp"
#include "psImplantParams.hpp"

#include "../psModelDb.hpp"

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace viennaps::modeldb {

inline std::string defaultAnnealCsvPath() {
  const auto &root = viennaps::getModelDbRoot();
  if (root.empty()) {
    throw ModelDbError(
        missingModelDataMessage(ModelDataKind::Anneal, "",
                                "The model DB root is not configured. Call "
                                "`viennaps::initModelDbRoot()` "
                                "or `viennaps::setModelDbRoot(path)` before "
                                "using model-DB anneal lookup."));
  }
  return root + "/anneal/annealing.csv";
}

inline const std::string &requireModelDbRoot(const ModelDataKind kind) {
  const auto &root = viennaps::getModelDbRoot();
  if (root.empty()) {
    throw ModelDbError(missingModelDataMessage(
        kind, "",
        "The model DB root is not configured. Call "
        "`viennaps::initModelDbRoot()` "
        "or `viennaps::setModelDbRoot(path)` before using model-DB lookup."));
  }
  return root;
}

template <typename NumericType, int D>
inline ImplantTableParams<NumericType>
lookupImplantTable(const std::string &species, const std::string &material,
                   const std::string &substrateType,
                   const NumericType energyKeV, const NumericType tiltDeg,
                   const NumericType rotationDeg, const NumericType dosePerCm2,
                   const NumericType screenThickness,
                   const NumericType damageLevel = NumericType(0)) {
  ImplantTableParams<NumericType> out;

  const auto sp = viennaps::modeldb::canonicalSpeciesToken(species);
  const auto mat = viennaps::modeldb::canonicalMaterialToken(material);
  const auto sub =
      substrateType.empty() ? std::string("amorphous") : substrateType;
  const auto &root = requireModelDbRoot(ModelDataKind::Implant);
  const auto speciesName = viennaps::modeldb::canonicalSpeciesName(species);
  const auto materialName = viennaps::modeldb::canonicalMaterialName(material);

  ImplantRecipe<NumericType> implantLookup;
  implantLookup.species = sp;
  implantLookup.material = mat;
  implantLookup.substrateType = sub;
  implantLookup.energyKeV = energyKeV;
  implantLookup.tiltDeg = tiltDeg;
  implantLookup.rotationDeg = rotationDeg;
  implantLookup.dosePerCm2 = dosePerCm2;
  implantLookup.screenThickness = screenThickness;
  implantLookup.damageLevel = damageLevel;
  implantLookup.useTableLookup = true;
  implantLookup.tableFileName = root + "/implant/" + speciesName + "_in_" +
                                materialName +
                                ((materialName == "silicon" && !sub.empty())
                                     ? "_" + viennaps::modeldb::lower(sub)
                                     : "") +
                                ".csv";

  out.implantRecipe = implantLookup;
  out.implantRecipe.useTableLookup = false;
  try {
    out.implantRecipe.entry =
        viennaps::tables::lookupImplantTableEntry<NumericType>(
            implantLookup.tableFileName, implantLookup.species,
            implantLookup.material, implantLookup.substrateType,
            implantLookup.energyKeV, implantLookup.tiltDeg,
            implantLookup.rotationDeg, implantLookup.dosePerCm2,
            implantLookup.screenThickness, implantLookup.damageLevel,
            implantLookup.preferredModel);
  } catch (const std::exception &error) {
    throw ModelDbError(missingModelDataMessage(
        ModelDataKind::Implant, implantLookup.tableFileName, error.what()));
  }

  DamageRecipe<NumericType> damageLookup;
  damageLookup.species = sp;
  damageLookup.material = mat;
  damageLookup.energyKeV = energyKeV;
  damageLookup.tiltDeg = tiltDeg;
  damageLookup.rotationDeg = rotationDeg;
  damageLookup.dosePerCm2 = dosePerCm2;
  damageLookup.screenThickness = screenThickness;
  damageLookup.useTableLookup = true;
  damageLookup.tableFileName =
      root + "/damage/" + speciesName + "_damage_in_" + materialName + ".csv";

  out.damageRecipe = damageLookup;
  out.damageRecipe.useTableLookup = false;
  try {
    out.damageRecipe.entry =
        viennaps::tables::lookupDamageTableEntry<NumericType>(
            damageLookup.tableFileName, damageLookup.species,
            damageLookup.material, damageLookup.energyKeV, damageLookup.tiltDeg,
            damageLookup.rotationDeg, damageLookup.dosePerCm2,
            damageLookup.screenThickness);
  } catch (const std::exception &error) {
    throw ModelDbError(missingModelDataMessage(
        ModelDataKind::Damage, damageLookup.tableFileName, error.what()));
  }

  return out;
}

template <typename NumericType>
inline NumericType evaluateArrhenius(const NumericType pref,
                                     const NumericType ea,
                                     const NumericType temperatureK) {
  constexpr NumericType kB = NumericType(8.617333262145e-5);
  const auto t = std::max(temperatureK, NumericType(1));
  return std::max(pref, NumericType(0)) *
         std::exp(-std::max(ea, NumericType(0)) / (kB * t));
}

template <typename NumericType>
inline NumericType
thermalAverageArrhenius(const NumericType pref, const NumericType ea,
                        const std::vector<NumericType> &durations,
                        const std::vector<NumericType> &temperatures,
                        const NumericType fallbackTemperatureK) {
  if (durations.empty() || temperatures.empty())
    return evaluateArrhenius(pref, ea, fallbackTemperatureK);

  NumericType total = 0;
  NumericType weighted = 0;
  if (temperatures.size() == durations.size()) {
    for (std::size_t i = 0; i < durations.size(); ++i) {
      const auto dt = std::max(durations[i], NumericType(0));
      total += dt;
      weighted += dt * evaluateArrhenius(pref, ea, temperatures[i]);
    }
  } else if (temperatures.size() == durations.size() + 1) {
    constexpr int samples = 16;
    for (std::size_t i = 0; i < durations.size(); ++i) {
      const auto dt = std::max(durations[i], NumericType(0));
      total += dt;
      NumericType segmentMean = 0;
      for (int j = 0; j < samples; ++j) {
        const auto a =
            (NumericType(j) + NumericType(0.5)) / NumericType(samples);
        const auto t =
            temperatures[i] * (NumericType(1) - a) + temperatures[i + 1] * a;
        segmentMean += evaluateArrhenius(pref, ea, t);
      }
      weighted += dt * segmentMean / NumericType(samples);
    }
  } else {
    return evaluateArrhenius(pref, ea, fallbackTemperatureK);
  }

  if (total <= NumericType(0))
    return evaluateArrhenius(pref, ea, fallbackTemperatureK);
  return weighted / total;
}

template <typename NumericType>
inline AnnealParams<NumericType>
lookupAnneal(const std::string &species, const std::string &substrateMaterial,
             const std::vector<NumericType> &durations,
             const std::vector<NumericType> &temperatures,
             const NumericType fallbackTemperatureK,
             const NumericType lengthUnitInCm,
             const std::string &annealCsvPath = "") {
  const auto unitCm = std::max(lengthUnitInCm, NumericType(1e-30));
  constexpr NumericType tableLengthUnitInCm = NumericType(1e-7); // nm
  const auto tableToCurrentLength = tableLengthUnitInCm / unitCm;
  const auto tableAreaToCurrentArea =
      tableToCurrentLength * tableToCurrentLength;
  const auto tableVolumeToCurrentVolume =
      tableAreaToCurrentArea * tableToCurrentLength;
  const auto cm3ConcentrationToCurrent = unitCm * unitCm * unitCm;
  const std::string dopantName = canonicalSpeciesName(species);
  const std::string substrateName = canonicalMaterialName(substrateMaterial);
  const std::string csvPath =
      annealCsvPath.empty() ? defaultAnnealCsvPath() : annealCsvPath;
  AnnealParams<NumericType> out;
  const auto thermalAverage = [&](const NumericType pref,
                                  const NumericType ea) {
    return thermalAverageArrhenius(pref, ea, durations, temperatures,
                                   fallbackTemperatureK);
  };
  const auto tableDiffusivity = [&](const NumericType pref,
                                    const NumericType ea) {
    return thermalAverage(pref * tableAreaToCurrentArea, ea);
  };
  const auto tableVolumeRate = [&](const NumericType pref,
                                   const NumericType ea) {
    return thermalAverage(pref * tableVolumeToCurrentVolume, ea);
  };
  const auto cm3Concentration = [&](const NumericType concentration) {
    return std::max(concentration, NumericType(0)) * cm3ConcentrationToCurrent;
  };

  NumericType dTotalPref = NumericType(-1), dTotalEa = NumericType(0);
  NumericType dIntPref = NumericType(-1), dIntEa = NumericType(0);
  NumericType dVacPref = NumericType(-1), dVacEa = NumericType(0);

  std::ifstream file(csvPath);
  if (!file.is_open()) {
    throw ModelDbError(missingModelDataMessage(
        ModelDataKind::Anneal, csvPath,
        "The calibrated annealing parameter table could not be opened."));
  }

  std::string line;
  while (std::getline(file, line)) {
    line = viennaps::modeldb::trim(line);
    if (line.empty() || line[0] == '#')
      continue;
    std::vector<std::string> cols;
    std::stringstream ss(line);
    std::string token;
    while (std::getline(ss, token, ','))
      cols.push_back(viennaps::modeldb::trim(token));
    if (cols.size() < 6)
      continue;

    const auto mat = viennaps::modeldb::lower(cols[0]);
    const auto cat = viennaps::modeldb::lower(cols[1]);
    const auto name = viennaps::modeldb::lower(cols[2]);
    const auto param = viennaps::modeldb::lower(cols[3]);
    if (mat != substrateName)
      continue;

    NumericType pref = NumericType(0), ea = NumericType(0);
    try {
      pref = static_cast<NumericType>(std::stod(cols[4]));
      ea = static_cast<NumericType>(std::stod(cols[5]));
    } catch (...) {
      continue;
    }

    if (cat == "dopant" && name == dopantName) {
      if (param == "d") {
        dTotalPref = pref;
        dTotalEa = ea;
      } else if (param == "int_d") {
        dIntPref = pref;
        dIntEa = ea;
      } else if (param == "vac_d") {
        dVacPref = pref;
        dVacEa = ea;
      } else if (param == "score_ifactor") {
        out.scoreIFactor = pref;
      } else if (param == "score_vfactor") {
        out.scoreVFactor = pref;
      } else if (param == "score_dfactor") {
        out.scoreDFactor = pref;
      } else if (param == "solubility") {
        out.solidSolubilityC0 = cm3Concentration(pref);
        out.solidSolubilityEa_eV = ea;
      }
    } else if (cat == "defect") {
      if (name == "interstitial") {
        if (param == "di")
          out.interstitialDiffusivity = tableDiffusivity(pref, ea);
        else if (param == "cstar") {
          out.interstitialEqC0 = cm3Concentration(pref);
          out.interstitialEqEa_eV = ea;
          out.enableDefectEquilibrium = true;
        }
      } else if (name == "vacancy") {
        if (param == "dv")
          out.vacancyDiffusivity = tableDiffusivity(pref, ea);
        else if (param == "cstar") {
          out.vacancyEqC0 = cm3Concentration(pref);
          out.vacancyEqEa_eV = ea;
          out.enableDefectEquilibrium = true;
        }
      } else if (name == "icluster") {
        if (param == "ikfi") {
          out.clusterKfi = thermalAverage(pref, ea);
          out.enableDefectClustering = true;
        } else if (param == "ikfc") {
          out.clusterKfc = tableVolumeRate(pref, ea);
          out.enableDefectClustering = true;
        } else if (param == "ikr") {
          out.clusterKr = thermalAverage(pref, ea);
          out.enableDefectClustering = true;
        } else if (param == "initpercent") {
          out.clusterInitFraction = pref / NumericType(100);
          out.enableDefectClustering = true;
        }
      }
    } else if (cat == "property") {
      if (param == "krec")
        out.defectRecombinationRate = tableVolumeRate(pref, ea);
    }
  }

  if (dTotalPref >= NumericType(0)) {
    out.dopantD0 = dTotalPref * tableAreaToCurrentArea;
    out.dopantEa_eV = dTotalEa;
  } else if (dIntPref >= NumericType(0) || dVacPref >= NumericType(0)) {
    if (dIntPref >= dVacPref) {
      out.dopantD0 = dIntPref * tableAreaToCurrentArea;
      out.dopantEa_eV = dIntEa;
    } else {
      out.dopantD0 = dVacPref * tableAreaToCurrentArea;
      out.dopantEa_eV = dVacEa;
    }
  } else {
    throw ModelDbError(missingModelDataMessage(
        ModelDataKind::Anneal, csvPath,
        "No dopant diffusivity entry was found for `" + dopantName + "` in `" +
            substrateName + "`."));
  }
  return out;
}

} // namespace viennaps::modeldb
