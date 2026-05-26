#pragma once

#include "psImplantProfile.hpp"
#include "psIonImplantation.hpp"

namespace viennaps {

template <typename NumericType> struct ImplantTableParams {
  ImplantRecipe<NumericType> implantRecipe;
  DamageRecipe<NumericType> damageRecipe;
};

template <typename NumericType, int D>
struct ImplantParams {
  SmartPointer<ImplantProfileModel<NumericType, D>> implantModel;
  SmartPointer<ImplantProfileModel<NumericType, D>> damageModel;
};

template <typename NumericType, int D>
inline void applyImplantParams(
    IonImplantation<NumericType, D> &implant,
    const ImplantTableParams<NumericType> &p) {
  implant.setImplantModel(
      SmartPointer<ImplantRecipeModel<NumericType, D>>::New(
          p.implantRecipe));
  implant.setDamageModel(
      SmartPointer<DamageRecipeModel<NumericType, D>>::New(
          p.damageRecipe));
}

template <typename NumericType, int D>
inline void applyImplantParams(
    IonImplantation<NumericType, D> &implant,
    const ImplantParams<NumericType, D> &p) {
  implant.setImplantModel(p.implantModel);
  if (p.damageModel)
    implant.setDamageModel(p.damageModel);
}

} // namespace viennaps
