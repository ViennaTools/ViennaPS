#pragma once

// Implant profile models for ViennaPS.
//
// This header imports the ViennaCS profile-model layer (depth + lateral
// concentration/damage profiles computed from Pearson IV moments or calibrated
// lookup tables) and re-exports the types in the viennaps namespace.
//
// The models are pure profile functions: given (depth, offset) they return a
// normalized concentration or damage density. They carry no ViennaPS process-
// framework dependency and are used by psIonImplantation and psAnneal.
//
// Dependencies resolved via the ViennaCS target; build ViennaPS with
//   -DVIENNAPS_VIENNACS_SOURCE=/path/to/ViennaCS_ionimplantation
// to use a local development branch instead of the published ViennaCS release.

#include <csImplantModel.hpp>
#include <models/csImplantDamage.hpp>
#include <models/csImplantPearson.hpp>
#include <models/csImplantTable.hpp>

#include "../materials/psMaterial.hpp"

#include <algorithm>
#include <string>

namespace viennaps {

// Abstract base: a normalized depth/lateral profile integrating to 1.
// getProfile(depth, offset) == getDepthProfile(depth) * getLateralProfile(offset, depth)
template <typename NumericType, int D>
using ImplantProfileModel = viennacs::ImplantModel<NumericType, D>;

// Pearson IV moment parameters { mu, sigma, gamma, beta }
template <typename NumericType>
using PearsonIVParameters = viennacs::constants::PearsonIVParameters<NumericType>;

// Lateral straggle model selector and its parameter bundle
using LateralStraggleModel = viennacs::LateralStraggleModel;
template <typename NumericType>
using LateralStraggleParameters = viennacs::LateralStraggleParameters<NumericType>;

// Single Pearson IV dopant depth profile + Gaussian lateral profile
template <typename NumericType, int D>
using ImplantPearsonIV = viennacs::ImplantPearsonIV<NumericType, D>;

// Pearson IV + exponential channeling tail
template <typename NumericType, int D>
using ImplantPearsonIVChanneling = viennacs::ImplantPearsonIVChanneling<NumericType, D>;

// Weighted sum of two Pearson IV components (amorphous head + channeling tail)
template <typename NumericType, int D>
using ImplantDualPearsonIV = viennacs::ImplantDualPearsonIV<NumericType, D>;

// Hobler-style damage depth profile (Gaussian + exponential bulk decay)
template <typename NumericType, int D>
using ImplantDamageHobler = viennacs::ImplantDamageHobler<NumericType, D>;

// Table-driven model types (data-file backed lookup + interpolation)
template <typename NumericType>
using ImplantTableEntry = viennacs::tables::ImplantTableEntry<NumericType>;

template <typename NumericType>
using ImplantRecipe = viennacs::tables::ImplantRecipe<NumericType>;

template <typename NumericType>
using DamageTableEntry = viennacs::tables::DamageTableEntry<NumericType>;

template <typename NumericType>
using DamageRecipe = viennacs::tables::DamageRecipe<NumericType>;

template <typename NumericType>
using ImplantTable = viennacs::tables::ImplantTable<NumericType>;

template <typename NumericType>
using DamageTable = viennacs::tables::DamageTable<NumericType>;

template <typename NumericType, int D>
using TableDrivenImplantModel = viennacs::tables::TableDrivenImplantModel<NumericType, D>;

template <typename NumericType, int D>
using RecipeDrivenImplantModel = viennacs::tables::RecipeDrivenImplantModel<NumericType, D>;

template <typename NumericType, int D>
using TableDrivenDamageModel = viennacs::tables::TableDrivenDamageModel<NumericType, D>;

template <typename NumericType, int D>
using RecipeDrivenDamageModel = viennacs::tables::RecipeDrivenDamageModel<NumericType, D>;

// Convenience accessor for the vsclib data root (set once at startup).
using viennacs::processutil::getVsclibRoot;
using viennacs::processutil::setVsclibRoot;

// Map a ViennaPS Material to the lowercase canonical name used in the ViennaCS
// implant data tables. Falls back to the built-in material name string for
// materials not explicitly listed.
[[nodiscard]] inline std::string implantMaterialName(Material material) {
  if (!material.isBuiltIn())
    return "unknown";

  switch (material.builtIn()) {
  case BuiltInMaterial::Si:
  case BuiltInMaterial::BulkSi:
  case BuiltInMaterial::aSi:
  case BuiltInMaterial::PolySi:
    return "silicon";
  case BuiltInMaterial::Ge:
    return "germanium";
  case BuiltInMaterial::SiGe:
    return "sige";
  case BuiltInMaterial::SiO2:
    return "oxide";
  case BuiltInMaterial::Si3N4:
  case BuiltInMaterial::SiN:
    return "nitride";
  default: {
    auto name = std::string(builtInMaterialToString(material.builtIn()));
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return name;
  }
  }
}

} // namespace viennaps
