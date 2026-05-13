#pragma once

#include "../materials/psMaterialMap.hpp"
#include "psAnnealParams.hpp"
#include "psImplantSetup.hpp"
#include "psModelDbLookup.hpp"

#include <algorithm>
#include <cctype>
#include <string>
#include <utility>
#include <vector>

namespace viennaps {

template <typename NumericType> struct AnnealModel {
  AnnealParams<NumericType> parameters;
  std::string source;
};

template <typename NumericType> struct AnnealSchedule {
  std::vector<NumericType> durations;
  std::vector<NumericType> temperatures;
};

template <typename NumericType>
inline NumericType
peakAnnealTemperature(const AnnealSchedule<NumericType> &schedule,
                      const NumericType fallback = NumericType(1323.15)) {
  if (schedule.temperatures.empty())
    return fallback;
  return *std::max_element(schedule.temperatures.begin(),
                           schedule.temperatures.end());
}

inline AnnealMode annealModeFromString(std::string value,
                                       const AnnealMode fallback) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  if (value == "implicit" || value == "gaussseidel" || value == "gauss-seidel")
    return AnnealMode::GaussSeidel;
  if (value == "explicit")
    return AnnealMode::Explicit;
  return fallback;
}

template <typename NumericType>
inline AnnealModel<NumericType> lookupAnneal(
    const std::string &dopantName, const std::string &substrateMaterial,
    const AnnealSchedule<NumericType> &schedule,
    const NumericType fallbackTemperatureK, const NumericType lengthUnitInCm) {
  AnnealModel<NumericType> out;
  out.parameters = modeldb::lookupAnneal<NumericType>(
      dopantName, substrateMaterial, schedule.durations, schedule.temperatures,
      fallbackTemperatureK, lengthUnitInCm);
  out.source = "model DB";
  return out;
}

template <typename NumericType, int D, typename ParameterSet>
inline AnnealModel<NumericType>
lookupAnneal(const ImplantSetup<NumericType, D, ParameterSet> &implantSetup,
             const AnnealSchedule<NumericType> &schedule,
             const NumericType fallbackTemperatureK,
             const Material substrateMaterial = Material::Si) {
  return lookupAnneal<NumericType>(
      implantSetup.species, MaterialMap::toString(substrateMaterial), schedule,
      fallbackTemperatureK, implantSetup.lengthUnitInCm);
}

template <typename NumericType>
inline AnnealModel<NumericType>
manualAnneal(const AnnealParams<NumericType> &parameters,
             std::string source = "manual parameters") {
  return {parameters, std::move(source)};
}

template <typename NumericType> struct AnnealSetup {
  AnnealModel<NumericType> model;
  AnnealSchedule<NumericType> schedule;
  NumericType peakTemperatureK = NumericType(1323.15);
  NumericType duration = NumericType(5);
  AnnealMode mode = AnnealMode::GaussSeidel;
  int implicitMaxIterations = 400;
  NumericType implicitTolerance = NumericType(1e-6);
  NumericType implicitRelaxation = NumericType(1);
  bool defectCoupling = true;
  DopantFields labels = dopantFields("dopant");
  Material substrateMaterial = Material::Si;
  std::vector<Material> diffusionMaterials = {Material::Si};
  std::vector<Material> blockingMaterials = {Material::Mask, Material::SiO2};
};

template <typename NumericType, int D>
inline void applyAnnealSetup(Anneal<NumericType, D> &anneal,
                             const AnnealSetup<NumericType> &config) {
  anneal.setSpeciesLabel(config.labels.total);
  anneal.setActiveLabel(config.labels.active);
  anneal.setDamageLabels(config.labels.damage, config.labels.lastDamage);
  anneal.setDefectLabels(config.labels.interstitial, config.labels.vacancy);
  anneal.setDefectClusterLabel(config.labels.cluster);
  anneal.setTemperature(config.peakTemperatureK);
  applyAnnealParams(anneal, config.model.parameters);
  anneal.setMode(config.mode);
  anneal.setImplicitSolverOptions(config.implicitMaxIterations,
                                  config.implicitTolerance,
                                  config.implicitRelaxation);

  if (!config.schedule.durations.empty() &&
      !config.schedule.temperatures.empty()) {
    anneal.setTemperatureSchedule(config.schedule.durations,
                                  config.schedule.temperatures);
  } else {
    anneal.setDuration(config.duration);
  }

  anneal.setDiffusionMaterials(config.diffusionMaterials);
  anneal.setBlockingMaterials(config.blockingMaterials);
  if (config.defectCoupling)
    anneal.enableDefectCoupling(true);
}

} // namespace viennaps
