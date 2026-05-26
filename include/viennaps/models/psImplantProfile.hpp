#pragma once

// Implant profile models for ViennaPS.
//
// This header defines the ViennaPS profile-model layer (depth + lateral
// concentration/damage profiles computed from Pearson IV moments or calibrated
// lookup tables) on top of the ViennaCS cell-set implant model interface.
//
// The models are pure profile functions: given (depth, offset) they return a
// normalized concentration or damage density. They carry no ViennaPS process-
// framework dependency and are consumed by psIonImplantation.
//
// Dependencies resolved via the ViennaCS target; build ViennaPS with
//   -DVIENNAPS_VIENNACS_SOURCE=/path/to/ViennaCS
// to use a local development branch instead of the published ViennaCS release.

#include <csImplantModel.hpp>

#include "psImplantConstants.hpp"
#include "psImplantDamage.hpp"
#include "psImplantPearson.hpp"
#include "psImplantTable.hpp"

#include "../materials/psMaterial.hpp"
#include <algorithm>
#include <string>

namespace viennaps {

// Abstract base: a normalized depth/lateral profile integrating to 1.
// getProfile(depth, offset) == getDepthProfile(depth) * getLateralProfile(offset, depth)
template <typename NumericType, int D>
using ImplantProfileModel = ImplantModel<NumericType, D>;

// Pearson IV moment parameters { mu, sigma, gamma, beta }
template <typename NumericType>
using PearsonIVParameters = viennaps::constants::PearsonIVParameters<NumericType>;

// Lateral straggle model selector and its parameter bundle
using LateralStraggleModel = viennaps::LateralStraggleModel;
template <typename NumericType>
using LateralStraggleParameters = viennaps::LateralStraggleParameters<NumericType>;

// Single Pearson IV dopant depth profile + Gaussian lateral profile
template <typename NumericType, int D>
using ImplantPearsonIV = viennaps::ImplantPearsonIV<NumericType, D>;

// Pearson IV + exponential channeling tail
template <typename NumericType, int D>
using ImplantPearsonIVChanneling = viennaps::ImplantPearsonIVChanneling<NumericType, D>;

// Weighted sum of two Pearson IV components (amorphous head + channeling tail)
template <typename NumericType, int D>
using ImplantDualPearsonIV = viennaps::ImplantDualPearsonIV<NumericType, D>;

// Hobler-style damage depth profile (Gaussian + exponential bulk decay)
template <typename NumericType, int D>
using ImplantDamageHobler = viennaps::ImplantDamageHobler<NumericType, D>;

// Table-driven model types (data-file backed lookup + interpolation)
template <typename NumericType>
using ImplantTableEntry = viennaps::tables::ImplantTableEntry<NumericType>;

template <typename NumericType>
using ImplantRecipe = viennaps::tables::ImplantRecipe<NumericType>;

template <typename NumericType>
using DamageTableEntry = viennaps::tables::DamageTableEntry<NumericType>;

template <typename NumericType>
using DamageRecipe = viennaps::tables::DamageRecipe<NumericType>;

template <typename NumericType>
using ImplantTable = viennaps::tables::ImplantTable<NumericType>;

template <typename NumericType>
using DamageTable = viennaps::tables::DamageTable<NumericType>;

template <typename NumericType, int D>
using ImplantTableModel = viennaps::tables::ImplantTableModel<NumericType, D>;

template <typename NumericType, int D>
using ImplantRecipeModel = viennaps::tables::ImplantRecipeModel<NumericType, D>;

template <typename NumericType, int D>
using DamageTableModel = viennaps::tables::DamageTableModel<NumericType, D>;

template <typename NumericType, int D>
using DamageRecipeModel = viennaps::tables::DamageRecipeModel<NumericType, D>;

// Map a ViennaPS Material to the lowercase canonical name used in the ViennaPS
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
